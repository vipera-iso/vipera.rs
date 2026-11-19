// Several activations against one row: how prefill stops paying for the weights per token.
//
// Same staging as the single-row kernel — one warp per row, the row copied into shared memory once —
// then each warp runs up to `MAX_BATCH` dots against that one copy. A chunk of eight tokens therefore
// reads the matrix once instead of eight times, which is the entire reason the CPU backend's prefill
// beats its decode by 6x.
//
// Layouts: `xq`, `xd`, `xs` hold `batch` activation sets of `blocks*256`, `blocks*8` and `blocks*8`;
// `y` holds `batch * rows` floats, entry-major.

#define QK_K 256
#define BLOCK_BYTES 144
#define MAX_BATCH 8

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

extern "C" __global__ void q4k_matvec_q8_batch(
    const unsigned char * __restrict__ w,
    const signed char * __restrict__ xq,
    const float * __restrict__ xd,
    const float * __restrict__ xs,
    float * __restrict__ y,
    int rows,
    int blocks_per_row,
    int batch)
{
    extern __shared__ unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * warps + warp;
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

    float acc[MAX_BATCH];
    for (int entry = 0; entry < MAX_BATCH; ++entry) {
        acc[entry] = 0.0f;
    }

    const int j = lane >> 2;
    const int eighth = lane & 3;
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char *blk = stage + b * BLOCK_BYTES;
        const float d = h2f(*(const unsigned short *)blk);
        const float dmin = h2f(*(const unsigned short *)(blk + 2));
        unsigned char sc, m;
        scale_min_k4(j, blk + 4, &sc, &m);

        const unsigned char *qs = blk + 16 + ((j >> 1) << 5) + eighth * 8;
        const unsigned int packed_lo = *(const unsigned int *)qs;
        const unsigned int packed_hi = *(const unsigned int *)(qs + 4);
        unsigned int q0, q1;
        if (j & 1) {
            q0 = (packed_lo >> 4) & 0x0F0F0F0Fu;
            q1 = (packed_hi >> 4) & 0x0F0F0F0Fu;
        } else {
            q0 = packed_lo & 0x0F0F0F0Fu;
            q1 = packed_hi & 0x0F0F0F0Fu;
        }
        const int g = b * 8 + j;
        const int at = b * QK_K + j * 32 + eighth * 8;
        for (int entry = 0; entry < batch; ++entry) {
            const int offset = entry * blocks_per_row * QK_K + at;
            const int goffset = entry * blocks_per_row * 8 + g;
            const int dot = __dp4a((int)q1, *(const int *)(xq + offset + 4),
                                   __dp4a((int)q0, *(const int *)(xq + offset), 0));
            const float da = xd[goffset];
            acc[entry] += d * (float)sc * da * (float)dot - dmin * (float)m * da * xs[goffset] * 0.25f;
        }
    }

    for (int entry = 0; entry < batch; ++entry) {
        for (int off = 16; off > 0; off >>= 1) {
            acc[entry] += __shfl_down_sync(0xffffffffu, acc[entry], off);
        }
        if (lane == 0) {
            y[entry * rows + row] = acc[entry];
        }
    }
}
