// A row of Q4_K weights against a float activation vector.
//
// Layout (144 bytes per 256 values): d (f16), dmin (f16), 12 packed 6-bit scales/mins, 128 packed
// 4-bit quants. Within a super-block, sub-block `j = v / 32` uses nible `j & 1` of `qs[32*(j/2) + v%32]`,
// which is the layout `dequantize_row_q4_K` in ggml walks.
//
// One warp owns one row: the row's bytes are staged into shared memory with a coalesced read (a
// thread-per-row read would stride 864 bytes and waste three quarters of every transaction), then the
// lanes split the 256 values of each super-block and reduce.
//
// No includes: the f16 conversion is open-coded so NVRTC needs no toolkit include path.

#define QK_K 256
#define BLOCK_BYTES 144
#define WARPS 8

__device__ __forceinline__ float h2f(unsigned short h) {
    const unsigned int sign = (h >> 15) & 1u;
    const unsigned int exp = (h >> 10) & 0x1fu;
    const unsigned int man = h & 0x3ffu;
    float v;
    if (exp == 0u) {
        v = ldexpf((float)man, -24);
    } else if (exp == 31u) {
        // Inf and NaN by bit pattern: no headers in this translation unit.
        v = man ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        v = ldexpf((float)(man | 0x400u), (int)exp - 25);
    }
    return sign ? -v : v;
}

__device__ __forceinline__ void scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void q4k_matvec(
    const unsigned char * __restrict__ w,
    const float * __restrict__ x,
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
        // 16 bytes per lane per step: a byte at a time issues a 32-byte transaction where the card
        // moves 128, and the staging read is the whole cost of this kernel.
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

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char *blk = stage + b * BLOCK_BYTES;
        const float d = h2f(*(const unsigned short *)blk);
        const float dmin = h2f(*(const unsigned short *)(blk + 2));
        const unsigned char *scales = blk + 4;
        const unsigned char *qs = blk + 16;
        const float *xb = x + b * QK_K;
        for (int v = lane; v < QK_K; v += 32) {
            const int j = v >> 5;
            unsigned char sc, m;
            scale_min_k4(j, scales, &sc, &m);
            const int qi = ((j >> 1) << 5) + (v & 31);
            const unsigned char q = (j & 1) ? (unsigned char)(qs[qi] >> 4) : (unsigned char)(qs[qi] & 0xF);
            acc += (d * (float)sc * (float)q - dmin * (float)m) * xb[v];
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}
