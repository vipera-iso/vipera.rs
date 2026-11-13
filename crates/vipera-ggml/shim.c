// ADR-001 shim: the only C in the tree that this engine calls.
//
// It exposes two pure functions and nothing else. It does not own state, threads or
// buffers, and it does not translate layouts beyond the block strides it is handed —
// the point of the bind is to call ggml's kernel, not to grow a second engine.

#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

// Declared in ggml/src/ggml-cpu/arch/x86/quants.c; the generic fallback has the same
// shape, so one declaration covers both.
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by, int nrc);

// Two ggml.c symbols are referenced by the quant source's row validation. Only the block
// sizes and names the K-quants path needs are answered; anything else aborts, so a wrong
// call cannot quietly produce a plausible row size.
size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q8_K: return sizeof(block_q8_K);
        case GGML_TYPE_F32:  return sizeof(float);
        default:             abort();
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return sizeof(block_q4_K) * (size_t) (ne / 256);
        case GGML_TYPE_F32:
            return sizeof(float) * (size_t) ne;
        default:
            abort();
    }
}

// The half and MX tables are only read by conversion and nvfp4/mxfp4 paths, none of which
// this bind calls. They exist to satisfy the linker and are left zeroed on purpose: if a
// future seam reaches one, the parity test fails on the value instead of the numbers being
// quietly wrong in the engine.
float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_Q8_K: return "q8_K";
        case GGML_TYPE_F32:  return "f32";
        default:             return "unknown";
    }
}

// ggml's assertion path lives in ggml.c, which is not part of this bind. Aborting is the
// same behaviour, and the shim never calls it in a valid measurement.
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

// IEEE half to single, bit for bit. ggml's macro is C++-only in this tree, and the table is
// the one thing the kernel genuinely needs from ggml_cpu_init().
static float fp16_to_fp32(uint16_t half) {
    const uint32_t sign = (uint32_t) (half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 0x1f;
    uint32_t mantissa = half & 0x3ff;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3ff;
            bits = sign | ((uint32_t) (127 - 15 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// ggml's x86 kernels read half scales through a 256 KB lookup table that ggml_cpu_init()
// fills in ggml-cpu.c. That file is not part of this bind, so the table is built here.
// Rust calls this once, before any kernel.
void vp_init_tables(void) {
    for (uint32_t bits = 0; bits < (1u << 16); ++bits) {
        ggml_table_f32_f16[bits] = fp16_to_fp32((uint16_t) bits);
    }
}

// Both return 0 on success. `cols` must be a multiple of 256.
int vp_quantize_row_q8_k(int64_t cols, const float * x, void * out) {
    if (cols <= 0 || cols % 256 != 0) {
        return 1;
    }
    quantize_row_q8_K_ref(x, (block_q8_K *) out, cols);
    return 0;
}

int vp_vec_dot_q4_k_q8_k(int64_t cols, const void * q4k_row, const void * q8k, float * out) {
    if (cols <= 0 || cols % 256 != 0) {
        return 1;
    }
    *out = 0.0f;
    ggml_vec_dot_q4_K_q8_K((int) cols, out, 0, q4k_row, 0, q8k, 0, 1);
    return 0;
}

// Repacked-weight entry points: the layout is `block_q4_Kx8`, eight rows interleaved
// per 256-column block. x86 kernels read it directly, which is what makes llama.cpp's
// prefill GEMM fast: no per-call unpack.
void ggml_quantize_mat_q8_K_4x8(const float * x, void * vy, int64_t k);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx,
                             const void * vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx,
                             const void * vy, int nr, int nc);

int vp_quantize_mat_q8_k_4x8(const float * x, void * out, int64_t cols) {
    if (cols <= 0 || cols % 256 != 0) {
        return 1;
    }
    ggml_quantize_mat_q8_K_4x8(x, out, cols);
    return 0;
}

int vp_gemv_q4_k_8x8_q8_k(int cols, float * out, const void * repacked,
                          const void * activations, int rows) {
    if (cols <= 0 || cols % 256 != 0 || rows <= 0 || rows % 8 != 0) {
        return 1;
    }
    // nr is the number of activation rows (one) and nc the number of weight rows.
    ggml_gemv_q4_K_8x8_q8_K(cols, out, 0, repacked, activations, 1, rows);
    return 0;
}

int vp_gemm_q4_k_8x8_q8_k(int cols, float * out, size_t stride, const void * repacked,
                          const void * activations, int rows, int columns) {
    if (cols <= 0 || cols % 256 != 0 || rows <= 0 || columns <= 0) {
        return 1;
    }
    ggml_gemm_q4_K_8x8_q8_K(cols, out, stride, repacked, activations, rows, columns);
    return 0;
}

// Build stamp, so a measurement can say which kernel it ran.
const char * vp_kernel_build(void) {
#if defined(__AVX512VNNI__)
    return "avx512vnni";
#elif defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}
