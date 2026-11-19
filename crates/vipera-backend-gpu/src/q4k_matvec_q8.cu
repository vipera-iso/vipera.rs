// The int8 path: Q4_K weights against Q8 activations, four products per instruction.
//
// Same tiling as the float kernel (one warp per row, the row staged in shared memory), but a value is
// no longer dequantized: the 4-bit quants stay packed and meet the int8 activations in `__dp4a`.
//
// Q4_K is an affine quantizer — a value is `d * sc_j * q - dmin * m_j` — so the sub-block contributes
//     d * sc_j * Σ q_i a_i  -  dmin * m_j * Σ a_i
// The first sum is the integer dot; the second is the sub-block's activation sum, which is why the
// host hands over `xs` alongside the scales. Both are folded per lane and reduced at the end.
//
// Lane → work: sub-block `j = lane >> 2`, the eighth of it `lane & 3`, so eight values per lane and two
// `dp4a` per super-block. The nibble that belongs to a sub-block is `j & 1`, its byte group `j >> 1`.

#define QK_K 256
#define BLOCK_BYTES 144
#define WARPS 8

__device__ __forceinline__ void scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float h2f(unsigned short h) {
    const unsigned int sign = (h >> 15) & 1u;
    const unsigned int exp = (h >> 10) & 0x1fu;
    const unsigned int man = h & 0x3ffu;
    float v;
    if (exp == 0u) {
        v = ldexpf((float)man, -24);
    } else if (exp == 31u) {
        v = man ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        v = ldexpf((float)(man | 0x400u), (int)exp - 25);
    }
    return sign ? -v : v;
}

extern "C" __global__ void q4k_matvec_q8(
    const unsigned char * __restrict__ w,
    const signed char * __restrict__ xq,
    const float * __restrict__ xd,
    const float * __restrict__ xs,
    float * __restrict__ y,
    int rows,
    int blocks_per_row)
{
    extern __shared__ unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS + warp;
    const int row_bytes = blocks_per_row * BLOCK_BYTES;
    unsigned char *stage = smem + warp * row_bytes;

    if (row < rows) {
        const uint4 *src = (const uint4 *)(w + (size_t)row * (size_t)row_bytes);
        uint4 *dst = (uint4 *)stage;
        const int quads = row_bytes >> 4;
        for (int i = lane; i < quads; i += 32) {
            dst[i] = src[i];
        }
    }
    __syncwarp();
    if (row >= rows) {
        return;
    }

    const int j = lane >> 2;          // sub-block inside the super-block
    const int eighth = lane & 3;      // which eight of its 32 values
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char *blk = stage + b * BLOCK_BYTES;
        const float d = h2f(*(const unsigned short *)blk);
        const float dmin = h2f(*(const unsigned short *)(blk + 2));
        unsigned char sc, m;
        scale_min_k4(j, blk + 4, &sc, &m);

        const unsigned char *qs = blk + 16 + ((j >> 1) << 5) + eighth * 8;
        const unsigned int packed_lo = *(const unsigned int *)qs;
        const unsigned int packed_hi = *(const unsigned int *)(qs + 4);
        // Four nibbles to four bytes: a mask on the whole word, which is two instructions instead of
        // eight shifts and ors.
        unsigned int q0, q1;
        if (j & 1) {
            q0 = (packed_lo >> 4) & 0x0F0F0F0Fu;
            q1 = (packed_hi >> 4) & 0x0F0F0F0Fu;
        } else {
            q0 = packed_lo & 0x0F0F0F0Fu;
            q1 = packed_hi & 0x0F0F0F0Fu;
        }

        const int at = b * QK_K + j * 32 + eighth * 8;
        const int a0 = *(const int *)(xq + at);
        const int a1 = *(const int *)(xq + at + 4);
        int dot = __dp4a((int)q0, a0, 0);
        dot = __dp4a((int)q1, a1, dot);

        const int g = b * 8 + j;
        const float da = xd[g];
        // Four lanes share a sub-block, so each carries a quarter of the activation-sum term.
        acc += d * (float)sc * da * (float)dot - dmin * (float)m * da * xs[g] * 0.25f;
    }

    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}
