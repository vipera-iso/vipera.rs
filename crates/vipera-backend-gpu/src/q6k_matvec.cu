// Q6_K against float activations.
//
// A Q4_K_M file is not all Q4_K: `attn_v` and `ffn_down` stay at 6 bits, so a layer on the device
// needs this too. 210 bytes per 256 values: 128 packed low nibbles, 64 packed high pairs, 16 signed
// sub-block scales, one f16 scale.
//
// Lane `l` in 0..31 owns eight values: the four quarter-rows `l`, `l+32`, `l+64`, `l+96` of each
// 128-value half — the same split `dequantize_row_q6_K` walks, so the index arithmetic is the
// reference's and not a reinvention.

#define QK_K 256
#define BLOCK_BYTES 210

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

extern "C" __global__ void q6k_matvec(
    const unsigned char * __restrict__ w,
    const float * __restrict__ x,
    float * __restrict__ y,
    int rows,
    int blocks_per_row)
{
    extern __shared__ unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * warps + warp;
    const int row_bytes = blocks_per_row * BLOCK_BYTES;
    unsigned char *stage = smem + warp * row_bytes;

    if (row < rows) {
        // Byte at a time: 210-byte blocks make a row an arbitrary length, so a row rarely starts on
        // a four- or sixteen-byte boundary and a `uint` staging read faults. Q6_K is the smaller
        // share of a layer, so the cheaper-to-be-correct load wins here.
        const unsigned char *src = w + (size_t)row * (size_t)row_bytes;
        for (int i = lane; i < row_bytes; i += 32) {
            stage[i] = src[i];
        }
    }
    __syncwarp();
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char *blk = stage + b * BLOCK_BYTES;
        const float d = h2f(*(const unsigned short *)(blk + 208));
        const float *xb = x + b * QK_K;
        for (int half = 0; half < 2; ++half) {
            const unsigned char *ql = blk + 64 * half;
            const unsigned char *qh = blk + 128 + 32 * half;
            const signed char *sc = (const signed char *)(blk + 192 + 8 * half);
            const int is = lane >> 4;
            const unsigned char low = ql[lane];
            const unsigned char high = qh[lane];
            const int q1 = (int)(low & 0xF) | (((int)(high >> 0) & 3) << 4);
            const int q2 = (int)(ql[lane + 32] & 0xF) | (((int)(high >> 2) & 3) << 4);
            const int q3 = (int)(low >> 4) | (((int)(high >> 4) & 3) << 4);
            const int q4 = (int)(ql[lane + 32] >> 4) | (((int)(high >> 6) & 3) << 4);
            const int base = half * 128;   // `xb` is already at this block
            acc += d * (float)sc[is + 0] * (float)(q1 - 32) * xb[base + lane];
            acc += d * (float)sc[is + 2] * (float)(q2 - 32) * xb[base + lane + 32];
            acc += d * (float)sc[is + 4] * (float)(q3 - 32) * xb[base + lane + 64];
            acc += d * (float)sc[is + 6] * (float)(q4 - 32) * xb[base + lane + 96];
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}
