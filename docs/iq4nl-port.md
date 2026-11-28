# IQ4_NL: codebook, block layout, kernel, and port effort

Status: **research only — no code changed**. Date: 2026-11-28.

Sources in this document:

- `IK` = `/home/user/llm-engines/ik_llama.cpp`, git HEAD `1180df7` ("Add clip op for Inkling2 (#2971)").
- `VIP` = this worktree (`vipera.rs`). Line numbers are from the files as they exist at the time of writing.

## 0. Bottom line

`IQ4_NL` is a 4.5 bpw **legacy** (32-value block) non-linear quant: GGML type id **20**, an 18-byte
block `{fp16 d, 16 bytes of packed 4-bit codes}`, and a fixed 16-entry `int8` lookup table. On disk it
is barely more complex than `Q4_0` — the only difference is the codebook instead of `nibble - 8`.

The performance story is the opposite of the storage story. In ik_llama.cpp the fast CPU path does
**not** run a native IQ4_NL integer dot: it LUT-decodes IQ4_NL into interleaved `Q8_0_R8`/`Q8_0_R16`
once and then runs the generic int8 matmul (`IK ggml/src/iqk/iqk_mul_mat.cpp:317-332`,
`iqk_gemm_legacy_quants.cpp:3906-3944`). The direct dot (`ggml-quants.c:12128`) aborts when the IQK
matmul is compiled in. So "port the kernel" really means "port a repack + reuse an int8 GEMM", which
is the same machinery `docs/design/repack-port-plan.md` already measured and closed for decode.

Consequence: a **reader port is small**; a **throughput-parity port is a new workstream with a fresh
measurement gate**, not a copy-paste.

## 1. Codebook

The codebook is a fixed, **block-independent** table of 16 `int8` levels
(`IK ggml/src/ggml-quants.c:3998`, `ggml/src/ggml-common.h:2652-2655`,
`ggml/src/ggml-cuda/common.cuh:435`, `gguf-py/gguf/quants.py:1212-1213`):

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| value | -127 | -104 | -83 | -65 | -49 | -35 | -22 | -10 | 1 | 13 | 25 | 38 | 53 | 69 | 89 | 113 |

Properties that matter for a port:

- It is **not symmetric** about zero: `min = -127`, `max = +113`, and there is no exact zero (the
  closest levels are `-10` and `+1`). Do not substitute a `Q4_0`-style `(nibble - 8)` decoder.
- It is the **first 16 entries of `iq4k_values[32]`**; the second 16 are used by IQ4_K / IQ4_XS
  families, which select a half with extra bits. `IQ4_NL` has no such selector bit: one table, one
  scale, no minimum.
- The scale `d` carries the sign and the magnitude: `d ≈ amax / 127` in the quantizer
  (`IK ggml/src/ggml-quants.c:15177-15215`, `15354-15375`). `d` is a raw `fp16`, read natively
  little-endian.

There are **two encodings of the same table** in ik, and mixing them up is the classic port bug:

| form | bytes | definition | used by |
|---|---|---|---|
| signed `int8` | `-127 … 113` | `kvalues_iq4nl` (`ggml-quants.c:3998`), `iq4k_values` (`ggml-common.h:2652`) | reference dequant/dot, CUDA dequant and MMQ LUT, gguf-py |
| biased `uint8` = signed + 128 | `1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241` | `load_iq4nl_values_128()` (`IK ggml/src/iqk/iqk_common.h:280-290`) | AVX2 `_mm256_shuffle_epi8` decode; decoded bytes are XORed with `0x80` (`-128`) to restore the signed value before storing into `Q8_0` |

`IQ4_NL_DequantizerU` uses the biased table; `IQ4_NL_DequantizerS` uses the signed one
(`IK ggml/src/iqk/iqk_gemm_legacy_quants.cpp:636-650`). For a read-only port only the signed table is
needed.

Quantization itself is not part of a reader port. For completeness: ik uses a scale search with
importance-matrix weights and `ntry = 7` (`ggml-quants.c:15177-15195`, `15354-15375`, AVX2 at
`14979`), and a per-type fudge factor from `ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_NL)`.

## 2. Block layout

```c
// IK ggml/src/ggml-common.h:721-726
#define QK4_NL 32
typedef struct {
    ggml_half d;            // 2 bytes, fp16 scale
    uint8_t   qs[QK4_NL/2]; // 16 bytes, 32 nibbles
} block_iq4_nl;             // 18 bytes  =>  18*8/32 = 4.5 bpw
```

- GGML type id **20** (`IK ggml/include/ggml.h:414`); ftype `GGML_FTYPE_MOSTLY_IQ4_NL = 19`
  (`ggml.h:551`), llama ftype 25 (`IK include/llama.h:183`).
- GGUF metadata: `(block_size = 32, type_size = 18)` (`IK gguf-py/gguf/constants.py:2983`).
- Type traits in `IK ggml/src/ggml.c:1345-1357`: `blck_size = 32`, `type_size = sizeof(block_iq4_nl)`,
  `vec_dot_type = GGML_Q8_0_VEC_DOT_TYPE`, `nrows = 1`, `row_meta_size = 0`. There is a
  `static_assert(QK4_NL == QK8_0)` at `ggml-quants.c:12138`: one 32-value activation block aligns with
  exactly one weight block.

Nibble order, from the reference dequantizer (`IK ggml/src/ggml-quants.c:4000-4013`):

```c
float d = GGML_FP16_TO_FP32(x[i].d);
for (int j = 0; j < 16; ++j) {
    y[j      ] = d * kvalues_iq4nl[x[i].qs[j] & 0xf]; // low nibble  -> first half
    y[j + 16 ] = d * kvalues_iq4nl[x[i].qs[j] >>  4]; // high nibble -> second half
}
```

So byte `j` holds weights `j` (bits 0-3) and `j+16` (bits 4-7); the 18-byte block is
`[d_lo, d_hi, q0 … q15]`. Rows must be a whole number of 32-value blocks, i.e. `cols % 32 == 0`;
`row_bytes = cols / 32 * 18`. This is the *same* byte count and scale placement as `Q4_0` — VIP's
`GgmlType::block_bytes` already says 18 for `Q4_0` — but VIP's CPU reader does not implement `Q4_0`
either (`VIP crates/vipera-backend-cpu/src/quant.rs:17-27`).

Two ik-only neighbours are **not** GGUF on-disk types and must not be added to a GGUF reader:

| type | id | layout | size | purpose |
|---|---|---|---|---|
| `IQ4_NL_R4` | 220 | `d[4]`, `qs[64]` | 72 B / 128 weights | 4-row interleaved repack (`ggml-common.h:727-731`) |
| `IQ4_NL_R8` | (internal) | `d[8]`, `qs[128]` | 144 B / 256 weights | 8-row interleaved repack (`ggml-common.h:732-736`) |

They are produced by `repack_iq4_nl` (`IK ggml/src/iqk/iqk_quantize.cpp:6041-6075`) and consumed only
inside the IQK matmul.

## 3. Kernel

### 3.1 Reference path (correctness oracle)

`ggml_vec_dot_iq4_nl_q8_0` (`IK ggml/src/ggml-quants.c:12128-12280`) computes, per 32-value block,

```
dot += fp16(d_w[b]) * fp16(d_x[b]) * Σ_i codebook(qs_i) * q8_i
```

It has NEON (`tbl` + `vdotq`), AVX2 (`pshufb` + `mul_add_epi8` + `madd_epi16`), AVX, POWER9 and
scalar tails, all doing the same LUT lookup. It begins with
`if (GGML_USE_IQK_MULMAT) GGML_ABORT("fatal error")` — in normal ik builds this function is *not* the
hot path.

### 3.2 Fast CPU path (what actually runs)

Dispatch chain:

1. `IK ggml/src/iqk/iqk_mul_mat.cpp:585` routes `IQ4_NL` to `iqk_convert_legacy_quants_q8_r8`;
   `iqk_mul_mat.cpp:1183` and `:1288` route it to `iqk_set_kernels_legacy_quants`.
2. `IK ggml/src/iqk/iqk_gemm_legacy_quants.cpp:3980-3989` selects `IQ4_NL_UnpackerU` on AVX-512,
   otherwise `IQ4_NL_UnpackerS` (`:866-873`). The dequantizers (`:636-650`) are a nibble unpack
   (`Dequantizer4bit`) plus one `_mm256_shuffle_epi8` table lookup.
3. `iqk_convert_qX_q80_r16` (`:3269-3301`, AVX-512) or `iqk_convert_qX_q80_r8` dequantizes **16 or 8
   rows at a time** into `block_q8_0_r16`/`block_q8_0_r8`, interleaving bytes so one vector load
   serves many rows. The biased LUT is XORed with `0x80` when writing Q8 bytes.
4. `iqk_mul_mat.cpp:317-332` chooses the conversion target: `Q8_0_R16` when ≥32 activation rows on
   AVX-512, `Q8_0_R8` otherwise; below that threshold the original type is kept.
5. The generic int8/Q8_0 matmul then consumes the repacked panel. The Q8_0 conversion for
   `Q4_0`/`Q5_0`/`Q6_0`/`IQ4_NL`/`MXFP4` is the pack of PR #2807 noted in `IK README.md:140,276`.

The arithmetic is therefore `Σ_b d_w[b] · d_x[b] · dot_i8(codebook_row, x_row)` — one scale product
per 32 values, no per-value float work. This is a *simpler* epilogue than Q4_K's six 6-bit subscales,
and it lines up with VIP's per-32 `QuantizedX` groups (`VIP quant.rs:166-173`).

### 3.3 GPU and other backends

- CUDA type traits: `qk = QK4_NL = 32`, `qr = QR4_NL = 2`, `qi = QI4_NL = 4`
  (`IK ggml/src/ggml-cuda/common.cuh:627-631`; constants at `ggml-common.h:143-144`).
- Scalar/GPU dequant: `dequantize_iq4_nl` (`ggml-cuda/dequantize.cuh:22-38`); table lookup codebook at
  `common.cuh:435`.
- MMVQ/MMQ dot: `vec_dot_iq4_nl_q8_1` with `VDR_IQ4_NL_Q8_1_MMVQ = 2`, `_MMQ = 4`
  (`ggml-cuda/vecdotq.cuh:1293-1322`), using `get_int_from_table_16` (`iqk_mmvq_templates.cuh:560`,
  AMD `v_perm_b32` path at `common.cuh:413`). Tile loader `load_tiles_iq4_nl`
  (`mmq.cuh:2119-2155`); traits bind it to the Q8_0 DP4A/MMA dot (`mmq.cuh:4440-4444`); the
  interleaved-ID variant is at `mmq_id_common.cuh:2738-2784`.
- Flash attention has a dedicated IQ4_NL K/V dequant path (`ggml-cuda/fattn-common.cuh:153,488`,
  `fattn-vec-common.cuh:136,487`), instantiated for head dim 128 (`fattn-vec-f16.cu:74-93`).
- Elsewhere: SYCL support is documented (`IK docs/backend/SYCL.md:53`), and Vulkan shaders define
  `QUANT_K_IQ4_NL 32`, `QUANT_R_IQ4_NL 2` (`vulkan-shaders/types.comp:1318-1340`) with dequant/copy
  implementations in `dequant_funcs*.comp`, `copy_to_quant.comp`, `mul_mm.comp`.

## 4. Port effort for vipera.rs

Estimates below are **judgement, not measured timings** — no code was written and no benchmark run.

### 4.1 Current VIP state

| layer | state | evidence |
|---|---|---|
| GGUF type registry | id 20 rejected | `VIP crates/vipera-core/src/gguf/types.rs:44-60` (`from_id`) |
| CPU block sizing | rejected | `VIP crates/vipera-backend-cpu/src/quant.rs:17-27` (`block()`) |
| CPU reader | only F32, F16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K | `quant.rs:873-980`, error text at `:49-53` |
| CPU dot | `dot_scalar` has a catch-all `_ => 0.0` | `quant.rs:609-644` — adding the enum variant without a dot arm would **silently return 0**, not an error |
| GPU backend | F32, F16, Q4_K, Q6_K only | `VIP crates/vipera-backend-gpu/src/model.rs:59-109` |
| Existing policy note | "anything belonging to i-quants (IQ2/IQ3/IQ4/trellis) cannot be ported" | `VIP docs/design/repack-port-plan.md:1,§5` — that was a **perf-value** verdict about kernel machinery, not a capability statement; do not read it as forbidding a reader |

### 4.2 Tier A — read it correctly (dequant + f32 dot)

Smallest useful port. Touch points:

1. `crates/vipera-core/src/gguf/types.rs`: add `IQ4_NL` variant, `from_id(20)`, `id() → 20`,
   `name() → "IQ4_NL"`, `block_size() → 32`, `block_bytes() → 18` (6 match arms).
2. `crates/vipera-backend-cpu/src/quant.rs`: add `(32, 18)` to `block()`; add a `read_row` arm using
   `blocks::<18>()`, `read_f16(block, 0)` and a `const CODEBOOK: [f32; 16]`, mirroring the loop in
   §2; add a `dot_scalar` arm (f32 activation path); update the `unsupported()` format list.
3. Leave `quantizes_for` unchanged (`quant.rs:185-192`): IQ4_NL needs no `QuantizedX` for Tier A, so
   `Activation.q` stays `None` and no activation-granularity decision (ADR-005) is triggered.
4. Tests, per `CONSTRAINTS.md` ("new behavior ships with a test that fails without it"): a
   `gguf_parse` case for id 20 (`VIP crates/vipera-core/tests/gguf_parse.rs`), a dequant parity test
   against fixed bytes generated by `IK gguf-py` (`quants.py:1212`) or `dequantize_row_iq4_nl`, and a
   small `matvec` regression. Include a negative value for `d` and both nibble halves in the vector.

No new dependency, no `unsafe` outside `simd.rs`, no on-disk layout, no ADR: the new type cannot
change output for any model that loads today. **Rough size: half a day to a day**, dominated by
fixture construction and review.

### 4.3 Tier B — integer-kernel parity with ik

This is a different kind of work. To match ik you must reproduce the **conversion + generic GEMM**,
not just add a LUT to `simd.rs`:

- Decode 8 or 16 IQ4_NL rows into an interleaved Q8 panel (`IQ4_NL_DequantizerU/S` plus
  `iqk_convert_qX_q80_r8/r16`), then reuse/extend VIP's Q8 int8 matmul. `unsafe` must stay in
  `crates/vipera-backend-cpu/src/simd.rs` (`CONSTRAINTS.md` EXC list).
- A simpler direct alternative is a nibble→i8 LUT plus `vpdpbusd` with VIP's existing per-32
  `QuantizedX` (`quant.rs:166-173`); the block boundaries already coincide, so **no ADR-005-style
  activation change is needed**. It is not what ik does, so it cannot inherit ik's throughput claim.
- The load-time repack is exactly mechanism M1 in `docs/design/repack-port-plan.md`, which is closed
  for decode on this machine (ik 8-row gemv 1.06x at 8T; llama's repack 0.74x). Any Tier B proposal
  must therefore run the `vipera-perf-loop` gates from scratch: kernel benchmark, ≥3% chunk median
  over ≥5 alternating pairs, bit/parity against the Tier A f32 path, before/after in
  `docs/benchmarks.md`. **Rough size: 3-5 days including measurement**, and it may legitimately end
  with "do not land".
- Do **not** add `IQ4_NL_R4`/`R8` ids (220/…) to the GGUF registry: they are runtime repacks.

### 4.4 Tier C — GPU

Independent of Tier A/B: `crates/vipera-backend-gpu/src/model.rs` would need a new block size/path,
a device dequant kernel (the CUDA pattern is §3.3), a parity test modelled on
`tests/q4k_parity.rs`, and a decision about flash attention. `cudarc` and the crate-wide `unsafe`
allow already exist for this crate (`CONSTRAINTS.md` EXC-006). **Rough size: 2-3 days for the matmul
path; flash attention is a further item.** No GPU claim should be made without a device measurement.

### 4.5 Failure modes to guard against in review

- Treating `d` as `f32`/`bf16` rather than `fp16`.
- Decoding high nibble first (ik decodes low → index `j`, high → `j+16`).
- Using the biased `+128` table as signed values, or XORing when the signed table is used.
- Confusing `IQ4_NL` with `IQ4_XS`/`IQ4_K` (256-value blocks, extra scale bits, second half of
  `iq4k_values`).
- Forgetting `dot_scalar`'s `_ => 0.0` fallback: the new variant must get an explicit arm.
- Accepting ik's internal repack ids as GGUF on-disk types.
- Adding a new dependency or an `unsafe` outside `simd.rs`.

### 4.6 Verification status

No implementation was performed, so build/test/clippy were not run for this change. Verify a future
port with `cargo test --workspace`, `cargo clippy --workspace --all-targets` and `cargo fmt --check`
(the checks named in `CONSTRAINTS.md`), plus the parity tests listed in §4.2.
