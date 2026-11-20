// The rest of one decoder layer, so the matvecs are not surrounded by host round trips.
//
// Every kernel here is a straight port of the CPU function of the same name in
// `vipera-backend-cpu/src/model.rs`; the comments name which one.

__device__ __forceinline__ void scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float h2f16(unsigned short h) {
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

// `rms_norm`: out = x / sqrt(mean(x^2) + eps) * weight. One block, one thread per value.
extern "C" __global__ void rmsnorm(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    float * __restrict__ out,
    int n,
    float epsilon)
{
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float inverse = rsqrtf(scratch[0] / (float)n + epsilon);
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inverse * weight[i];
    }
}

// `rope_with`: NeoX layout, pairs (i, i + head_dim/2), one block per head.
extern "C" __global__ void rope(
    float * __restrict__ vector,
    int position,
    int heads,
    int head_dim,
    float base)
{
    const int head = blockIdx.x;
    const int half = head_dim >> 1;
    if (head >= heads) {
        return;
    }
    float *v = vector + head * head_dim;
    for (int index = threadIdx.x; index < half; index += blockDim.x) {
        const float exponent = 2.0f * (float)index / (float)head_dim;
        const float inv = 1.0f / powf(base, exponent);
        float sine, cosine;
        sincosf((float)position * inv, &sine, &cosine);
        const float low = v[index];
        const float high = v[index + half];
        v[index] = low * cosine - high * sine;
        v[index + half] = low * sine + high * cosine;
    }
}

// `attend`: causal grouped-query attention for one new position. One block per query head, K and V
// read from the cache rows this block's KV head owns.
extern "C" __global__ void attention_decode(
    const float * __restrict__ q,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    float * __restrict__ out,
    int positions,
    int heads,
    int kv_heads,
    int head_dim,
    float scale)
{
    extern __shared__ float shared[];   // [positions] scores, then [head_dim/32] dot partials
    float *scores = shared;
    float *partials = shared + positions;

    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int kv_row = kv_heads * head_dim;
    const float *qh = q + head * head_dim;

    for (int at = 0; at < positions; at = at + 1) {
        const float *kh = k_cache + (size_t)at * kv_row + kv_head * head_dim;
        float dot = qh[tid] * kh[tid];
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        }
        if ((tid & 31) == 0) {
            partials[tid >> 5] = dot;
        }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int warp = 0; warp < (int)(blockDim.x >> 5); ++warp) {
                total += partials[warp];
            }
            scores[at] = total * scale;
        }
        __syncthreads();
    }

    // Softmax over the whole history.
    float largest = __int_as_float(0xff800000);   // -inf: no headers in this translation unit
    for (int at = tid; at < positions; at += blockDim.x) {
        largest = fmaxf(largest, scores[at]);
    }
    partials[tid] = largest;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] = fmaxf(partials[tid], partials[tid + stride]);
        }
        __syncthreads();
    }
    const float top = partials[0];
    __syncthreads();
    float sum = 0.0f;
    for (int at = tid; at < positions; at += blockDim.x) {
        scores[at] = expf(scores[at] - top);
        sum += scores[at];
    }
    partials[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        __syncthreads();
    }
    const float total = partials[0];
    __syncthreads();

    float acc = 0.0f;
    for (int at = 0; at < positions; ++at) {
        const float *vh = v_cache + (size_t)at * kv_row + kv_head * head_dim;
        acc += scores[at] / total * vh[tid];
    }
    out[head * head_dim + tid] = acc;
}

// `attend_shared`: the same causal attention, with one block per *KV* head instead of per query
// head, so the `group` query heads that share a K/V row walk the cache together.
//
// The per-head kernel above makes every query head in a group re-read the same `positions *
// head_dim` of K and V; at 4k positions on a 40-head/8-KV-head model that is ~168 MB per layer of
// cache traffic for ~33 MB of distinct rows, and the card's L2 cannot hold either. Here a block owns
// one KV head: its first `head_dim` threads load the K row once into shared memory, every head in the
// group scores it from there, and the V row is read by the group's warps with the same index, so the
// L1 serves one row instead of `group` of them.
//
// Scores are the only thing that leaves the block, because `group * positions` floats do not fit in
// shared memory at long contexts (80 KB at 4096 for a group of five). They go to a scratch buffer the
// layer owns, one row per query head: 80 KB of traffic against the MBs the shares save.
extern "C" __global__ void attention_decode_shared(
    const float * __restrict__ q,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    float * __restrict__ out,
    float * __restrict__ scores,
    int positions,
    int heads,
    int kv_heads,
    int head_dim,
    float scale)
{
    // One slot per warp of every head in the group, plus the K row and the two per-head scalars.
    const int group = heads / kv_heads;
    const int warps = head_dim / 32;
    extern __shared__ float shared[];
    float *kh = shared;                              // [head_dim]
    float *partials = kh + head_dim;                 // [group * warps]
    float *head_max = partials + group * warps;      // [group]
    float *head_sum = head_max + group;              // [group]

    const int kv_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int local = tid / head_dim;                // which query head of the group
    const int d = tid % head_dim;
    const int warp = tid >> 5;
    const int across = tid & 31;
    const int head = kv_head * group + local;
    const int kv_row = kv_heads * head_dim;
    const float *qh = q + head * head_dim;
    const size_t base = (size_t)kv_head * head_dim + d;
    // One score row per query head: `heads * positions`, not `group * positions`, because every KV
    // block writes its own group's rows.
    float *mine = scores + (size_t)(kv_head * group + local) * positions;

    for (int at = 0; at < positions; ++at) {
        if (local == 0) {
            kh[d] = k_cache[(size_t)at * kv_row + base];
        }
        __syncthreads();
        float dot = qh[d] * kh[d];
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        }
        if (across == 0) {
            partials[warp] = dot;
        }
        __syncthreads();
        if (d == 0) {
            float total = 0.0f;
            for (int w = 0; w < warps; ++w) {
                total += partials[local * warps + w];
            }
            mine[at] = total * scale;
        }
        __syncthreads();   // the next iteration overwrites `kh`
    }

    // Softmax over the whole history, per head, across the head's `head_dim` threads.
    float largest = __int_as_float(0xff800000);
    for (int at = d; at < positions; at += head_dim) {
        largest = fmaxf(largest, mine[at]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        largest = fmaxf(largest, __shfl_down_sync(0xffffffffu, largest, off));
    }
    if (across == 0) {
        partials[warp] = largest;
    }
    __syncthreads();
    if (d == 0) {
        float top = __int_as_float(0xff800000);
        for (int w = 0; w < warps; ++w) {
            top = fmaxf(top, partials[local * warps + w]);
        }
        head_max[local] = top;
    }
    __syncthreads();
    const float top = head_max[local];

    float sum = 0.0f;
    for (int at = d; at < positions; at += head_dim) {
        const float weight = expf(mine[at] - top);
        mine[at] = weight;
        sum += weight;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if (across == 0) {
        partials[warp] = sum;
    }
    __syncthreads();
    if (d == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) {
            total += partials[local * warps + w];
        }
        head_sum[local] = 1.0f / total;
    }
    __syncthreads();
    const float inverse = head_sum[local];

    // Every head of the group indexes the same V row, so the rows the block reads once are the rows
    // the whole group multiplies by: that is the traffic this kernel exists to remove.
    float acc = 0.0f;
    for (int at = 0; at < positions; ++at) {
        acc += mine[at] * inverse * v_cache[(size_t)at * kv_row + base];
    }
    out[head * head_dim + d] = acc;
}

// `swiglu`: out = silu(gate) * up.
extern "C" __global__ void swiglu(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float * __restrict__ out,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// `add_assign`: a += b, the residual and the QKV biases.
extern "C" __global__ void add_inplace(
    float * __restrict__ a,
    const float * __restrict__ b,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += b[i];
    }
}

// `QuantizedX::new`: one block per group of 32, `scale = max|x| / 127`.
extern "C" __global__ void quantize_q8(
    const float * __restrict__ x,
    signed char * __restrict__ q,
    float * __restrict__ d,
    float * __restrict__ s,
    int groups)
{
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    if (group >= groups) {
        return;
    }
    const float *chunk = x + group * 32;
    __shared__ float scratch[32];
    float value = lane < 32 ? chunk[lane] : 0.0f;
    scratch[lane] = fabsf(value);
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        }
        __syncthreads();
    }
    const float largest = scratch[0];
    const float scale = largest == 0.0f ? 0.0f : largest / 127.0f;
    if (lane == 0) {
        d[group] = scale;
    }
    int rounded = 0;
    if (scale != 0.0f) {
        const float scaled = value / scale;
        rounded = (int)roundf(fminf(fmaxf(scaled, -127.0f), 127.0f));
        q[group * 32 + lane] = (signed char)rounded;
    } else {
        q[group * 32 + lane] = 0;
    }
    scratch[lane] = (float)rounded;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        s[group] = scratch[0];
    }
}

// One row of Q4_K weights as f32: the token embedding lookup, which a matvec would compute by
// multiplying 151,936 rows by a one-hot vector.
extern "C" __global__ void q4k_row(
    const unsigned char * __restrict__ row,
    float * __restrict__ out,
    int cols)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned char *blk = row + b * 144;
    const float d = h2f16(*(const unsigned short *)blk);
    const float dmin = h2f16(*(const unsigned short *)(blk + 2));
    for (int v = tid; v < 256; v += blockDim.x) {
        const int j = v >> 5;
        unsigned char sc, m;
        scale_min_k4(j, blk + 4, &sc, &m);
        const int qi = ((j >> 1) << 5) + (v & 31);
        const unsigned char q = (j & 1) ? (unsigned char)(blk[16 + qi] >> 4) : (unsigned char)(blk[16 + qi] & 0xF);
        out[b * 256 + v] = d * (float)sc * (float)q - dmin * (float)m;
    }
}
