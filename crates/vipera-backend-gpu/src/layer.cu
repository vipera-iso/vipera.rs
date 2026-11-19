// The rest of one decoder layer, so the matvecs are not surrounded by host round trips.
//
// Every kernel here is a straight port of the CPU function of the same name in
// `vipera-backend-cpu/src/model.rs`; the comments name which one.

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
