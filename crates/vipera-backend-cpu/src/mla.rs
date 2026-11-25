//! Multi-head latent attention (MLA), in the absorbed form a latent cache is fast in.
//!
//! DeepSeek-V2/V3 attention does not keep a key and a value per head. Per position it keeps one
//! latent vector `c` of `kv_lora_rank` values plus one rotary key of `qk_rope_head_dim`, and
//! rebuilds the per-head pair with two learned up-projections:
//!
//! ```text
//! k_nope[h] = W_UK[h] * c          v[h] = W_UV[h] * c
//! ```
//!
//! Read back the obvious way, each position costs a `nope_dim`-wide matvec per head before it can
//! be scored, so the latent cache saves bandwidth and pays it back in arithmetic. The kernel
//! FlashMLA runs instead is the **absorbed** one, which folds both projections out of the
//! per-position walk (`ggml-cuda/latent_attn.cu` in llama.cpp):
//!
//! ```text
//! score[j] = (W_UK[h]ᵀ q_nope[h]) · c[j] + q_rope[h] · k_rope[j]
//! out[h]   = W_UV[h] * Σ_j p_j c[j] / Σ_j p_j
//! ```
//!
//! The query is projected *down* into the latent space once per query, so a score is a dot over
//! `rank` values (512) against the cache instead of over `nope_dim` (128) against a rebuilt key,
//! and the value sum accumulates `rank` values per head instead of `value_dim`: one `W_UV` matvec
//! per query, never per position. What the absorbed form is really buying is that the cache row a
//! key tile decodes is shared by every head, so the head-ratio traffic disappears from the walk.
//!
//! That CUDA kernel takes the first of the two projections as already folded into the query by the
//! graph above it: its cache rows hold 576 values (512 latent + 64 rotary), one shared key row
//! per position, and the value GEMM reads the latent slice in place. This module folds `W_UK`
//! inside the kernel because it is handed the un-absorbed query, which is the same arithmetic one
//! step earlier; `W_UV` is applied after the value sum in both.
//!
//! The tile is the shape the existing `attend_rows_shared` kernel uses, and for the same reason:
//! one decode of a key tile feeds every query in the tile. Two GEMMs run over the latent — scores
//! as `Q_abs[rows][rank] * Cᵀ[rank][keys]` plus the same shape over the rotary half, values as
//! `P[rows][keys] * C[keys][rank]` — and the softmax is the online one, so a query's history needs
//! no second pass and no score buffer the size of the history.
//!
//! # Two arms, and why the switch exists
//!
//! `VIPERA_MLA=1` selects the absorbed kernel; off (the default) selects
//! [`attend_decompressed`], which materialises the whole layer's K and V and runs plain attention
//! over them. The arms agree within rounding, not bit for bit: absorbing reorders the products (a
//! dot over the latent space instead of over the head space) and folds the value projection after
//! the sum instead of before it. The test module holds them to each other and to a plain-attention
//! oracle, which is what this repo asks of a regrouped path (`VIPERA_ATTN_GEMM` is the same
//! arrangement, and for the same reason: it stays off until the owner says otherwise).
//!
//! # What is not here
//!
//! This module is the kernel, not an architecture. `model`'s forward pass implements the qwen2,
//! qwen3 and llama block layouts, whose caches hold one `f16` key and value row per KV head and
//! whose GGUFs carry no `kv_lora_rank` latent or `W_UK`/`W_UV` tensors, so nothing on those models
//! gives the absorbed form something to read and the switch has no consumer in the engine yet.
//! Wiring it needs a DeepSeek-V2/V3 block layout — the latent cache, the two up-projections and
//! the rope split — which is a separate change: routing a qwen2 layer through this module would
//! mean up-projecting a latent that does not exist.

use vipera_errors::ViperaError;

/// Query rows one tile handles. The accumulator is `rank` wide per row, so the tile has to stay
/// small enough that one head's `QUERY_TILE * rank` state lives in cache rather than being
/// streamed. `attend_rows_shared` uses the same height for its `head_dim`-wide accumulators.
const QUERY_TILE: usize = 32;

/// Key positions one tile decodes: one decode of a `KEY_TILE * rank` latent block feeds both
/// GEMMs of every query in the tile.
const KEY_TILE: usize = 64;

/// The shape of one MLA layer's attention.
///
/// `rank` is the model's `kv_lora_rank` (the latent width the cache stores), `nope_dim` its
/// `qk_nope_head_dim`, `rope_dim` its `qk_rope_head_dim`, and `value_dim` its `v_head_dim`. The
/// key and value widths are per head and need not be equal, which is why they are two fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaShape {
    /// Query heads. Every head reads the one latent vector.
    pub heads: usize,
    /// Latent values per position, `kv_lora_rank`.
    pub rank: usize,
    /// Values per head in the non-rotary key, `qk_nope_head_dim`.
    pub nope_dim: usize,
    /// Values per head in the rotary key, `qk_rope_head_dim`.
    pub rope_dim: usize,
    /// Values per head in the output, `v_head_dim`.
    pub value_dim: usize,
}

impl MlaShape {
    /// Values one query head occupies in the query tensor: its non-rotary then its rotary part.
    fn query_width(&self) -> usize {
        self.nope_dim + self.rope_dim
    }

    /// The score scale, `1 / sqrt(nope_dim + rope_dim)`.
    ///
    /// A score sums over both halves of the key, so the width being normalised is their sum (192
    /// for DeepSeek-V2's 128 + 64 per head), not `nope_dim` alone.
    pub fn score_scale(&self) -> f32 {
        1.0 / ((self.nope_dim + self.rope_dim) as f32).sqrt()
    }
}

/// The per-head up-projections the latent cache leaves out.
///
/// Both are head-major with one row per output value, which is how a `GGUF` holds a `[rank,
/// width]` tensor: `key_up[(head * nope_dim + d) * rank + r]` is `W_UK[h][d][r]`.
#[derive(Debug, Clone, Copy)]
pub struct MlaWeights<'a> {
    /// `[heads][value_dim][rank]`: `v[h] = value_up[h] * c`.
    pub value_up: &'a [f32],
    /// `[heads][nope_dim][rank]`: `k_nope[h] = key_up[h] * c`.
    pub key_up: &'a [f32],
}

/// The latent cache: what a position costs instead of one key and one value per head.
#[derive(Debug, Clone, Copy)]
pub struct MlaCache<'a> {
    /// `[positions][rank]`, shared by every head.
    pub latent: &'a [f32],
    /// `[positions][rope_dim]`, the rotary key half, already rotated.
    pub rope: &'a [f32],
}

/// Causal MLA attention over `positions.len()` queries, dispatched by `VIPERA_MLA`.
///
/// `query` holds each query's heads back to back in `nope_dim + rope_dim` values (the `q_nope ||
/// q_rope` layout DeepSeek's `q_proj` produces), `positions[entry]` is how many keys that entry
/// sees (every position below its own), and `out` gets `value_dim` values per head in the same
/// head order. The rotary half of `query` and every `cache.rope` row must already be rotated.
///
/// # Errors
///
/// [`ViperaError::invalid_argument`] when a slice is too short for the shape it is indexed with;
/// neither arm reads past an extent it was given.
pub fn attend(
    query: &[f32],
    cache: &MlaCache<'_>,
    positions: &[usize],
    shape: MlaShape,
    weights: &MlaWeights<'_>,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let extents = Extents::check(query, cache, positions, shape, weights, out)?;
    if extents.history == 0 {
        // Every entry's history is empty: the answer is zeros, not whatever the caller's buffer
        // held. `out` was checked long enough for exactly this many values.
        out[..positions.len() * shape.heads * shape.value_dim].fill(0.0);
        return Ok(());
    }
    if absorbed_enabled() {
        attend_absorbed(query, cache, positions, shape, weights, out);
    } else {
        attend_decompressed(query, cache, positions, shape, weights, out, extents);
    }
    Ok(())
}

/// Whether the absorbed latent kernel runs: `VIPERA_MLA`, default off.
///
/// The same acceptance as `VIPERA_ATTN_GEMM`: `1`, `true`, `on` or `yes`. Off is the arm that
/// materialises K and V, so leaving the switch alone keeps the arithmetic a layer had before the
/// latent cache existed.
fn absorbed_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VIPERA_MLA").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    })
}

/// The sizes both arms index with, checked once so neither has to re-check them.
#[derive(Debug, Clone, Copy)]
struct Extents {
    /// The longest history any entry sees: both walks stop here.
    history: usize,
}

impl Extents {
    fn check(
        query: &[f32],
        cache: &MlaCache<'_>,
        positions: &[usize],
        shape: MlaShape,
        weights: &MlaWeights<'_>,
        out: &mut [f32],
    ) -> Result<Self, ViperaError> {
        let entries = positions.len();
        let history = positions.iter().copied().max().unwrap_or(0);
        let short = |what: &str, have: usize, need: usize| {
            ViperaError::invalid_argument(format!(
                "MLA {what} holds {have} values, and {entries} entries of this shape need {need}"
            ))
        };
        if shape.heads == 0 || shape.value_dim == 0 {
            return Err(ViperaError::invalid_argument(
                "MLA needs at least one head and one value per head",
            ));
        }
        if shape.rank == 0 {
            return Err(ViperaError::invalid_argument(
                "MLA needs a non-zero latent rank",
            ));
        }
        if shape.query_width() == 0 {
            return Err(ViperaError::invalid_argument(
                "MLA needs at least one key dimension (nope_dim or rope_dim)",
            ));
        }
        let query_need = entries * shape.heads * shape.query_width();
        if query.len() < query_need {
            return Err(short("query", query.len(), query_need));
        }
        let out_need = entries * shape.heads * shape.value_dim;
        if out.len() < out_need {
            return Err(short("output", out.len(), out_need));
        }
        let latent_need = history * shape.rank;
        if cache.latent.len() < latent_need {
            return Err(short("latent cache", cache.latent.len(), latent_need));
        }
        let rope_need = history * shape.rope_dim;
        if cache.rope.len() < rope_need {
            return Err(short("rope cache", cache.rope.len(), rope_need));
        }
        let key_need = shape.heads * shape.nope_dim * shape.rank;
        if weights.key_up.len() < key_need {
            return Err(short("key up-projection", weights.key_up.len(), key_need));
        }
        let value_need = shape.heads * shape.value_dim * shape.rank;
        if weights.value_up.len() < value_need {
            return Err(short(
                "value up-projection",
                weights.value_up.len(),
                value_need,
            ));
        }
        Ok(Self { history })
    }
}

/// The absorbed kernel: scores and values straight out of the latent cache, two GEMMs per tile.
///
/// The head loop is outside the key-tile loop because the online softmax state is per head, and
/// the transposed latent block is rebuilt inside it because holding every head's accumulator for
/// a tile at once is 8 MiB at 128 heads of rank 512. The transpose writes `rank * KEY_TILE` values
/// against the score GEMM's `QUERY_TILE * rank * KEY_TILE` products, so it is ~3% and is not worth
/// hoisting at the cost of that footprint.
fn attend_absorbed(
    query: &[f32],
    cache: &MlaCache<'_>,
    positions: &[usize],
    shape: MlaShape,
    weights: &MlaWeights<'_>,
    out: &mut [f32],
) {
    let entries = positions.len();
    // An entry that sees no keys reads as zeros. Zeroing the answer up front is what makes that
    // hold for a whole tile that is skipped as well as for one row of a tile that runs.
    out[..entries * shape.heads * shape.value_dim].fill(0.0);
    let MlaShape {
        heads,
        rank,
        nope_dim,
        rope_dim,
        value_dim,
    } = shape;
    let width = shape.query_width();
    let scale = shape.score_scale();

    // Per query tile: the query projected into the latent space, the online state of each row, and
    // the latent accumulator the value GEMM folds into.
    let mut projected = vec![0.0_f32; QUERY_TILE * rank];
    let mut m = [f32::NEG_INFINITY; QUERY_TILE];
    let mut d = [0.0_f32; QUERY_TILE];
    let mut acc = vec![0.0_f32; QUERY_TILE * rank];
    let mut scores = vec![0.0_f32; QUERY_TILE * KEY_TILE];
    // One key tile with the latent and the rotary keys transposed for the score GEMM's `B`
    // operand: `[rank][keys]` and `[rope_dim][keys]`. The value GEMM reads the latent in its
    // cached order, so only the score side needs a copy.
    let mut latent_t = vec![0.0_f32; rank * KEY_TILE];
    let mut rope_t = vec![0.0_f32; rope_dim * KEY_TILE];

    for tile in (0..entries).step_by(QUERY_TILE) {
        let rows = (tile + QUERY_TILE).min(entries) - tile;
        let tile_positions = &positions[tile..tile + rows];
        let tile_history = tile_positions.iter().copied().max().unwrap_or(0);
        if tile_history == 0 {
            continue;
        }
        for head in 0..heads {
            // `q_abs[r] = Σ_d q_nope[d] * W_UK[h][d][r]`: the absorption, once per query and head.
            for row in 0..rows {
                let q = &query[((tile + row) * heads + head) * width..][..nope_dim];
                let slot = &mut projected[row * rank..][..rank];
                slot.fill(0.0);
                for (dim, value) in q.iter().enumerate() {
                    let up = &weights.key_up[(head * nope_dim + dim) * rank..][..rank];
                    crate::simd::axpy_f32(slot, *value, up);
                }
            }

            m[..rows].fill(f32::NEG_INFINITY);
            d[..rows].fill(0.0);
            acc[..rows * rank].fill(0.0);

            let mut start = 0;
            while start < tile_history {
                let keys = KEY_TILE.min(tile_history - start);
                let latent = &cache.latent[start * rank..][..keys * rank];
                for at in 0..rank {
                    for key in 0..keys {
                        latent_t[at * keys + key] = latent[key * rank + at];
                    }
                }
                for at in 0..rope_dim {
                    for key in 0..keys {
                        rope_t[at * keys + key] = cache.rope[(start + key) * rope_dim + at];
                    }
                }

                // Both score GEMMs accumulate into the same tile: a score's latent part and its
                // rotary part are one sum, split by which operand carries them.
                scores[..rows * keys].fill(0.0);
                crate::simd::gemm(
                    rows,
                    rank,
                    keys,
                    &projected,
                    rank,
                    &latent_t[..rank * keys],
                    &mut scores[..rows * keys],
                    keys,
                );
                if rope_dim > 0 {
                    crate::simd::gemm(
                        rows,
                        rope_dim,
                        keys,
                        &query[tile * heads * width + head * width + nope_dim..],
                        heads * width,
                        &rope_t[..rope_dim * keys],
                        &mut scores[..rows * keys],
                        keys,
                    );
                }

                for row in 0..rows {
                    // A row's visible part of the tile is a prefix, since positions only grow
                    // along a chunk: the rest is its future and must weigh nothing.
                    let visible = keys.min(tile_positions[row].saturating_sub(start));
                    let row_scores = &mut scores[row * keys..][..keys];
                    row_scores[visible..].fill(0.0);
                    if visible == 0 {
                        continue;
                    }
                    for weight in &mut row_scores[..visible] {
                        *weight *= scale;
                    }
                    let block_max = row_scores[..visible]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let largest = m[row].max(block_max);
                    let alpha = (m[row] - largest).exp();
                    if alpha != 1.0 {
                        d[row] *= alpha;
                        for value in &mut acc[row * rank..][..rank] {
                            *value *= alpha;
                        }
                    }
                    m[row] = largest;
                    d[row] += crate::simd::exp_sum_shifted(&mut row_scores[..visible], largest);
                }

                // The value GEMM: every row folds the tile's latent rows at its softmax weights.
                crate::simd::gemm(
                    rows,
                    keys,
                    rank,
                    &scores[..rows * keys],
                    keys,
                    latent,
                    &mut acc[..rows * rank],
                    rank,
                );
                start += keys;
            }

            // `out[h] = W_UV[h] * acc / d`: one up-projection per query and head, after the sum.
            for row in 0..rows {
                let base = ((tile + row) * heads + head) * value_dim;
                let dst = &mut out[base..][..value_dim];
                if d[row] == 0.0 {
                    // Every tile was skipped: the row saw nothing.
                    continue;
                }
                let inverse = 1.0 / d[row];
                let row_acc = &acc[row * rank..][..rank];
                let up = &weights.value_up[head * value_dim * rank..];
                for (value, up_row) in dst.iter_mut().zip(up.chunks_exact(rank)) {
                    *value = crate::simd::dot_f32(row_acc, up_row) * inverse;
                }
            }
        }
    }
}

/// The arm the switch turns off: rebuild the whole layer's K and V, then plain attention.
///
/// This is what a latent cache looks like before absorption, and it is the oracle the absorbed
/// kernel is tested against. It materialises `positions * heads * nope_dim` keys and
/// `positions * heads * value_dim` values — 268 MiB of keys alone at 128 heads of 128 over 4k
/// positions, against the 8 MiB the same history costs as latent rows — which is the trade MLA
/// makes and the reason the absorbed arm is the one behind the switch.
fn attend_decompressed(
    query: &[f32],
    cache: &MlaCache<'_>,
    positions: &[usize],
    shape: MlaShape,
    weights: &MlaWeights<'_>,
    out: &mut [f32],
    extents: Extents,
) {
    let history = extents.history;
    let MlaShape {
        heads,
        rank,
        nope_dim,
        rope_dim,
        value_dim,
    } = shape;
    let width = shape.query_width();
    let scale = shape.score_scale();

    let mut keys = vec![0.0_f32; history * heads * nope_dim];
    let mut values = vec![0.0_f32; history * heads * value_dim];
    for position in 0..history {
        let latent = &cache.latent[position * rank..][..rank];
        for head in 0..heads {
            let up = &weights.key_up[head * nope_dim * rank..];
            let key = &mut keys[(position * heads + head) * nope_dim..][..nope_dim];
            for (value, up_row) in key.iter_mut().zip(up.chunks_exact(rank)) {
                *value = crate::simd::dot_f32(latent, up_row);
            }
            let up = &weights.value_up[head * value_dim * rank..];
            let value = &mut values[(position * heads + head) * value_dim..][..value_dim];
            for (slot, up_row) in value.iter_mut().zip(up.chunks_exact(rank)) {
                *slot = crate::simd::dot_f32(latent, up_row);
            }
        }
    }

    let mut scores = vec![0.0_f32; history.max(1)];
    for (entry, &seen) in positions.iter().enumerate() {
        let q = &query[entry * heads * width..];
        for head in 0..heads {
            let q_nope = &q[head * width..][..nope_dim];
            let q_rope = &q[head * width + nope_dim..][..rope_dim];
            let mut best = f32::NEG_INFINITY;
            for (position, score) in scores.iter_mut().enumerate().take(seen) {
                let key = &keys[(position * heads + head) * nope_dim..][..nope_dim];
                let rope = &cache.rope[position * rope_dim..][..rope_dim];
                *score = (crate::simd::dot_f32(q_nope, key) + crate::simd::dot_f32(q_rope, rope))
                    * scale;
                best = best.max(*score);
            }
            let dst = &mut out[(entry * heads + head) * value_dim..][..value_dim];
            dst.fill(0.0);
            if seen == 0 {
                continue;
            }
            let mut total = 0.0_f32;
            for score in scores.iter_mut().take(seen) {
                *score = crate::model::exp(*score - best);
                total += *score;
            }
            let inverse = 1.0 / total;
            for (position, score) in scores.iter().take(seen).enumerate() {
                let value = &values[(position * heads + head) * value_dim..][..value_dim];
                crate::simd::axpy_f32(dst, score * inverse, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic spread of values in `[-0.5, 0.5)`: the tests compare kernels against each
    /// other, so the inputs must be reproducible and wide enough that a reordered product differs.
    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / 16_777_216.0 - 0.5
        }

        fn fill(&mut self, count: usize) -> Vec<f32> {
            (0..count).map(|_| self.next()).collect()
        }
    }

    /// Two heads, rank 8, five values per head: small enough that the transposes and the tiles are
    /// exercised without the test spending its time inside the GEMMs.
    const SHAPE: MlaShape = MlaShape {
        heads: 2,
        rank: 8,
        nope_dim: 6,
        rope_dim: 4,
        value_dim: 5,
    };

    /// Query, cache and weights of one case, generated once so every arm sees the same bytes.
    struct Case {
        query: Vec<f32>,
        latent: Vec<f32>,
        rope: Vec<f32>,
        key_up: Vec<f32>,
        value_up: Vec<f32>,
        positions: Vec<usize>,
    }

    impl Case {
        fn new(shape: MlaShape, positions: Vec<usize>, seed: u32) -> Self {
            let mut rng = Lcg(seed);
            let entries = positions.len();
            let history = positions.iter().copied().max().unwrap_or(0);
            Self {
                query: rng.fill(entries * shape.heads * shape.query_width()),
                latent: rng.fill(history * shape.rank),
                rope: rng.fill(history * shape.rope_dim),
                key_up: rng.fill(shape.heads * shape.nope_dim * shape.rank),
                value_up: rng.fill(shape.heads * shape.value_dim * shape.rank),
                positions,
            }
        }

        fn cache(&self) -> MlaCache<'_> {
            MlaCache {
                latent: &self.latent,
                rope: &self.rope,
            }
        }

        fn weights(&self) -> MlaWeights<'_> {
            MlaWeights {
                value_up: &self.value_up,
                key_up: &self.key_up,
            }
        }

        /// One arm's output for the case. The arms are called directly so a test can pick one,
        /// which is what `VIPERA_MLA` does through [`attend`].
        fn run(&self, shape: MlaShape, absorbed: bool) -> Vec<f32> {
            let mut out = vec![f32::NAN; self.positions.len() * shape.heads * shape.value_dim];
            if absorbed {
                attend_absorbed(
                    &self.query,
                    &self.cache(),
                    &self.positions,
                    shape,
                    &self.weights(),
                    &mut out,
                );
            } else {
                let extents = Extents {
                    history: self.positions.iter().copied().max().unwrap_or(0),
                };
                attend_decompressed(
                    &self.query,
                    &self.cache(),
                    &self.positions,
                    shape,
                    &self.weights(),
                    &mut out,
                    extents,
                );
            }
            out
        }
    }

    /// The largest per-value difference between two outputs of the same case.
    fn largest_gap(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .fold(0.0_f32, |gap, (l, r)| gap.max((l - r).abs()))
    }

    /// Plain causal attention over K and V the test builds from the up-projections: the arithmetic
    /// the decompressed arm has to reproduce, written without either arm's walks.
    fn oracle(case: &Case, shape: MlaShape, entry: usize, seen: usize) -> Vec<f32> {
        let rank = shape.rank;
        let width = shape.query_width();
        let scale = shape.score_scale();
        let query = &case.query[entry * shape.heads * width..];
        let mut out = vec![0.0_f32; shape.heads * shape.value_dim];
        for head in 0..shape.heads {
            let mut scores = vec![0.0_f32; seen];
            for (position, score) in scores.iter_mut().enumerate() {
                let c = &case.latent[position * rank..][..rank];
                let mut total = 0.0_f32;
                for dim in 0..shape.nope_dim {
                    let up = &case.key_up[(head * shape.nope_dim + dim) * rank..][..rank];
                    total += query[head * width + dim] * crate::simd::dot_f32(c, up);
                }
                for dim in 0..shape.rope_dim {
                    total += query[head * width + shape.nope_dim + dim]
                        * case.rope[position * shape.rope_dim + dim];
                }
                *score = total * scale;
            }
            let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - best).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (position, weight) in weights.iter().enumerate() {
                let c = &case.latent[position * rank..][..rank];
                for dim in 0..shape.value_dim {
                    let up = &case.value_up[(head * shape.value_dim + dim) * rank..][..rank];
                    out[head * shape.value_dim + dim] +=
                        weight / total * crate::simd::dot_f32(c, up);
                }
            }
        }
        out
    }

    #[test]
    fn the_absorbed_kernel_matches_the_decompressed_one_within_rounding() {
        let case = Case::new(SHAPE, vec![1, 2, 3, 4, 5], 7);
        let absorbed = case.run(SHAPE, true);
        let decompressed = case.run(SHAPE, false);
        assert!(absorbed.iter().all(|value| value.is_finite()));
        assert!(
            absorbed.iter().any(|value| value.abs() > 1e-3),
            "the case produced only zeros, so it proves nothing"
        );
        let gap = largest_gap(&absorbed, &decompressed);
        assert!(gap < 1e-5, "the arms differ by {gap}");
    }

    #[test]
    fn the_decompressed_arm_is_plain_attention_over_the_up_projections() {
        let case = Case::new(SHAPE, vec![5], 11);
        let decompressed = case.run(SHAPE, false);
        let expected = oracle(&case, SHAPE, 0, 5);
        let gap = largest_gap(&decompressed, &expected);
        assert!(gap < 1e-6, "the arm differs from the oracle by {gap}");
    }

    #[test]
    fn history_longer_than_a_key_tile_keeps_the_online_softmax_in_step() {
        // 100 positions cross a `KEY_TILE` boundary, and the causal prefixes leave the tail of the
        // last tile invisible to three of the four entries.
        let shape = SHAPE;
        let case = Case::new(shape, vec![1, 33, 65, 100], 13);
        let absorbed = case.run(shape, true);
        let decompressed = case.run(shape, false);
        let gap = largest_gap(&absorbed, &decompressed);
        assert!(gap < 1e-5, "the arms differ by {gap} after two tiles");

        // Both arms could agree while sharing a misread prefix, so every entry is also held to the
        // oracle over exactly the keys it may see.
        let width = shape.heads * shape.value_dim;
        for (entry, &seen) in case.positions.iter().enumerate() {
            let rows = &absorbed[entry * width..(entry + 1) * width];
            let expected = oracle(&case, shape, entry, seen);
            let gap = largest_gap(rows, &expected);
            assert!(
                gap < 1e-5,
                "entry {entry} ({seen} keys) is {gap} off the oracle"
            );
        }
    }

    #[test]
    fn an_entry_whose_history_is_one_key_is_that_key_value() {
        let shape = SHAPE;
        let case = Case::new(shape, vec![1], 17);
        let absorbed = case.run(shape, true);
        let latent = &case.latent[..shape.rank];
        for head in 0..shape.heads {
            for dim in 0..shape.value_dim {
                let up =
                    &case.value_up[(head * shape.value_dim + dim) * shape.rank..][..shape.rank];
                let expected = crate::simd::dot_f32(latent, up);
                let got = absorbed[head * shape.value_dim + dim];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "head {head} value {dim} came back {got}, not {expected}"
                );
            }
        }
    }

    #[test]
    fn an_entry_with_no_history_writes_zeros_and_does_not_poison_its_neighbours() {
        let shape = SHAPE;
        let case = Case::new(shape, vec![0, 2, 0, 3], 19);
        let absorbed = case.run(shape, true);
        let decompressed = case.run(shape, false);
        let width = shape.heads * shape.value_dim;
        for entry in [0, 2] {
            assert!(
                absorbed[entry * width..(entry + 1) * width]
                    .iter()
                    .all(|value| *value == 0.0),
                "entry {entry} saw nothing but did not read as zeros"
            );
        }
        assert!(largest_gap(&absorbed, &decompressed) < 1e-5);

        // A case with no keys at all never enters a tile, so both arms have to write the zeros
        // themselves rather than leave the caller's buffer as it was.
        let empty = Case::new(shape, vec![0], 31);
        for arm in [true, false] {
            assert!(
                empty.run(shape, arm).iter().all(|value| *value == 0.0),
                "the {arm} arm left a no-history answer unset"
            );
        }
    }

    #[test]
    fn a_layer_without_a_rope_half_still_matches() {
        let shape = MlaShape {
            rope_dim: 0,
            ..SHAPE
        };
        let case = Case::new(shape, vec![1, 4, 9], 23);
        let absorbed = case.run(shape, true);
        let decompressed = case.run(shape, false);
        let gap = largest_gap(&absorbed, &decompressed);
        assert!(gap < 1e-5, "the arms differ by {gap} without rope");
    }

    #[test]
    fn a_slice_too_short_for_the_shape_is_refused_by_name() {
        let case = Case::new(SHAPE, vec![3], 29);
        let mut out = vec![0.0_f32; 3 * SHAPE.heads * SHAPE.value_dim];
        let cache = MlaCache {
            latent: &case.latent[..SHAPE.rank - 1],
            rope: &case.rope,
        };
        let error = attend(
            &case.query,
            &cache,
            &case.positions,
            SHAPE,
            &case.weights(),
            &mut out,
        )
        .expect_err("a short latent cache was accepted");
        let message = format!("{error}");
        assert!(
            message.contains("latent cache"),
            "the error names the wrong slice: {message}"
        );
    }

    #[test]
    fn zero_length_and_single_token_shapes_do_not_panic_and_slice_kv_correctly() {
        // No queries: nothing to attend, no keys to read.
        let shape = SHAPE;
        let empty = Case::new(shape, vec![], 41);
        let mut out = vec![f32::NAN; 0];
        attend(
            &empty.query,
            &empty.cache(),
            &empty.positions,
            shape,
            &empty.weights(),
            &mut out,
        )
        .expect("zero entries should be accepted");
        assert!(out.is_empty());

        // A single token seeing a single key is the value up-projection of that key.
        let one = Case::new(shape, vec![1], 43);
        let absorbed = one.run(shape, true);
        let decompressed = one.run(shape, false);
        let expected = oracle(&one, shape, 0, 1);
        assert!(largest_gap(&absorbed, &expected) < 1e-6);
        assert!(largest_gap(&decompressed, &expected) < 1e-6);

        // Single-token odd shapes: no nope half, no rope half, and a rank-1 latent.
        let nope_only = MlaShape {
            rope_dim: 0,
            ..shape
        };
        let case = Case::new(nope_only, vec![1], 47);
        let gap = largest_gap(&case.run(nope_only, true), &case.run(nope_only, false));
        assert!(gap < 1e-5, "no-rope odd shape differed by {gap}");

        let rope_only = MlaShape {
            nope_dim: 0,
            ..shape
        };
        let case = Case::new(rope_only, vec![1, 2], 53);
        let absorbed = case.run(rope_only, true);
        let decompressed = case.run(rope_only, false);
        assert!(largest_gap(&absorbed, &decompressed) < 1e-5);
        for (entry, &seen) in case.positions.iter().enumerate() {
            let width = rope_only.heads * rope_only.value_dim;
            let got = &absorbed[entry * width..(entry + 1) * width];
            let want = oracle(&case, rope_only, entry, seen);
            assert!(
                largest_gap(got, &want) < 1e-5,
                "nope_dim=0 entry {entry} off oracle by {}",
                largest_gap(got, &want)
            );
        }

        let rank_one = MlaShape { rank: 1, ..shape };
        let case = Case::new(rank_one, vec![1, 3, 2], 59);
        let absorbed = case.run(rank_one, true);
        let decompressed = case.run(rank_one, false);
        assert!(largest_gap(&absorbed, &decompressed) < 1e-5);
    }

    #[test]
    fn degenerate_shapes_are_refused_without_panicking() {
        let case = Case::new(SHAPE, vec![1], 61);

        let rank_zero = MlaShape { rank: 0, ..SHAPE };
        let mut out = vec![0.0_f32; rank_zero.heads * rank_zero.value_dim];
        let query = vec![0.0_f32; rank_zero.heads * rank_zero.query_width()];
        let err = attend(
            &query,
            &MlaCache {
                latent: &[],
                rope: &case.rope[..rank_zero.rope_dim],
            },
            &[1],
            rank_zero,
            &MlaWeights {
                value_up: &[],
                key_up: &[],
            },
            &mut out,
        )
        .expect_err("rank=0 should be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("rank"),
            "rank=0 error should name the degenerate dimension: {msg}"
        );

        let no_key_dims = MlaShape {
            nope_dim: 0,
            rope_dim: 0,
            ..SHAPE
        };
        let mut out = vec![0.0_f32; no_key_dims.heads * no_key_dims.value_dim];
        let err = attend(
            &[],
            &MlaCache {
                latent: &case.latent[..no_key_dims.rank],
                rope: &[],
            },
            &[1],
            no_key_dims,
            &MlaWeights {
                value_up: &case.value_up
                    [..no_key_dims.heads * no_key_dims.value_dim * no_key_dims.rank],
                key_up: &[],
            },
            &mut out,
        )
        .expect_err("no key dimensions should be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("nope_dim") || msg.contains("rope_dim") || msg.contains("key"),
            "no-key-dims error should name the degenerate dimension: {msg}"
        );
    }
}
