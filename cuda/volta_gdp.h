// Gate activations and device helpers the gated dual projection units share.
#pragma once

#include <cuda_fp16.h>
#include <climits>
#include <cstdint>

#define WARP 32

enum class GdpAct { kSigmoid, kSwish };

template <GdpAct A>
__device__ __forceinline__ float gdp_act(float g) {
    // ftz exp and approximate divide: MUFU.EX2 + MUFU.RCP, no denormal fixup, no IEEE divide subroutine
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-1.4426950408889634f * g));
    const float s = __fdividef(1.0f, 1.0f + e);
    return A == GdpAct::kSwish ? g * s : s;
}

// Instantiate the launch once per activation; kAct names it inside.
#define GDP_ACT_DISPATCH(a, ...)                                                       \
    switch (a) {                                                                       \
    case 0: {                                                                          \
        constexpr GdpAct kAct = GdpAct::kSigmoid;                                      \
        __VA_ARGS__;                                                                   \
    } break;                                                                           \
    case 1: {                                                                          \
        constexpr GdpAct kAct = GdpAct::kSwish;                                        \
        __VA_ARGS__;                                                                   \
    } break;                                                                           \
    default:                                                                           \
        return ffi::Error::InvalidArgument("gdp: activation is 0 (sigmoid) or 1 (swish), got " + \
                                           std::to_string(a));                         \
    }

// Every fused kernel runs four warps. ln_gdp and the projections give each warp 16 rows of a 64-row
// tile; the tri-mul epilogue holds two input tiles, thus 32 rows and a 2x2 warp grid.
constexpr int kThreads = 128;
constexpr int kRows = 64;
constexpr int kEpiRows = 32;
// Weights stream through shared memory as [kTileN][kTileK] tiles of a transposed [N][K] matrix.
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kWsLd = kTileK + 8;
// The per-head attention bias is a row reduction; a dot would pad 4-8 heads to a tensor-core tile.
constexpr int kMaxHeads = 8;
// The transition's hidden activation stays in shared memory, kHidden columns at a time.
constexpr int kHidden = 64;

struct LnGdpArgs {
    const __half *x, *wpT, *bp, *wgT, *bg, *mask; // x [M,K], wpT/wgT [P,K], bp/bg [P], mask [M]
    const float *scale, *offset;                  // [K]
    __half *left, *right;                         // [P/2, M] each
    int M, P;
    float eps;
};

struct EpilogueArgs {
    const float* x_cm;                     // [C, M] triangle product, channel major
    const __half *x, *woT, *bo, *wgT, *bg; // x [M,CA] raw module input, woT [O,C], wgT [O,CA]
    const __half* keep;                    // [M] or null, see tile_work
    const float *in_scale, *in_offset;     // [CA]
    const float *ln_scale, *ln_offset;     // [C]
    __half* out;                           // [M, O]
    int M, O;
    float eps;
};

struct ProjArgs {
    const __half *x, *wT, *b, *residual,
        *wbT;                    // x [M,K], wT [N,K], b [N], residual [M,N], wbT [h,K]
    const __half* keep;          // [M] or null, see tile_work
    const float *scale, *offset; // [K]
    __half *out, *bias;          // out [M,N], bias [h,M]
    int M, N, heads;
    float eps;
};

struct TransitionArgs {
    const __half *x, *w1T, *b1, *w2T,
        *b2;                     // x [M,C], w1T [H,C], b1 [H], w2T [H/kHidden][C][kHidden]
    const __half* keep;          // [M] or null, see tile_work
    const float *scale, *offset; // [C]
    __half* out;                 // [M, C]
    int M, H;
    float eps;
};

// What a row tile does, as the Pallas kernels' unless_padding: compute it when a row of keep is above
// zero, write what padding gives (zeros, or nothing where the output aliases the residual) when its
// rows are all zero, nothing at all when they are all below zero (outside what any consumer reads).
enum TileWork { kCompute, kPadding, kSkip };

// Every thread of the block must call it. Rows past M read as other; null keep computes every tile.
template <int R>
__device__ __forceinline__ TileWork tile_work(const __half* __restrict__ keep, int M, int m0,
                                              float other) {
    if (keep == nullptr) {
        return kCompute;
    }
    bool live = false, zero = false;
    for (int r = threadIdx.x; r < R; r += kThreads) {
        const float k = m0 + r < M ? __half2float(keep[m0 + r]) : other;
        live |= k > 0.f;
        zero |= k == 0.f;
    }
    if (__syncthreads_or(live)) {
        return kCompute;
    }
    return __syncthreads_or(zero) ? kPadding : kSkip;
}

// Zeros in rows [m0, m0+R) of a row-major [M][N] matrix, N a multiple of 8.
template <int R>
__device__ __forceinline__ void zero_rows(__half* __restrict__ out, int N, int M, int m0) {
    const int V = N / 8;
    for (int i = threadIdx.x; i < R * V; i += kThreads) {
        const int r = i / V;
        if (m0 + r < M) {
            reinterpret_cast<uint4*>(out + (long long)(m0 + r) * N)[i - r * V] =
                make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// Zeros in columns [m0, m0+R) of a channel-major [C][M] matrix.
template <int R>
__device__ __forceinline__ void zero_columns(__half* __restrict__ out, int C, int M, int m0) {
    for (int i = threadIdx.x; i < R * C; i += kThreads) {
        const int c = i / R, r = i - c * R;
        if (m0 + r < M) {
            out[(long long)c * M + m0 + r] = __float2half(0.f);
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// Copy rows [r0, r0+R) x columns [c0, c0+C) of a row-major matrix with ld columns into shared
// [R][S_LD], 16 bytes at a time. Rows at or past rmax read as zero.
template <int R, int C, int S_LD>
__device__ __forceinline__ void load_rows(__half* s, const __half* __restrict__ g, int ld, int r0,
                                          int c0, int rmax) {
    constexpr int V = C / 8;
    static_assert(C % 8 == 0 && S_LD % 8 == 0, "rows must be whole 16-byte vectors");
    for (int i = threadIdx.x; i < R * V; i += kThreads) {
        const int r = i / V, c = (i - r * V) * 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r0 + r < rmax) {
            v = *reinterpret_cast<const uint4*>(g + (long long)(r0 + r) * ld + c0 + c);
        }
        *reinterpret_cast<uint4*>(s + r * S_LD + c) = v;
    }
}

// LayerNorm the R rows of s [R][S_LD] in place, a warp per row, two-pass fp32 statistics.
template <int R, int C, int S_LD>
__device__ __forceinline__ void ln_rows(__half* s, const float* __restrict__ scale,
                                        const float* __restrict__ offset, float eps) {
    constexpr int V = C / WARP;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    float sc[V], of[V];
#pragma unroll
    for (int i = 0; i < V; ++i) {
        sc[i] = scale[lane + i * WARP];
        of[i] = offset[lane + i * WARP];
    }
    for (int r = warp; r < R; r += kThreads / WARP) {
        __half* row = s + r * S_LD;
        float v[V], sum = 0.f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            v[i] = __half2float(row[lane + i * WARP]);
            sum += v[i];
        }
        const float mean = warp_sum(sum) * (1.f / C);
        float sq = 0.f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            v[i] -= mean;
            sq += v[i] * v[i];
        }
        const float rstd = rsqrtf(warp_sum(sq) * (1.f / C) + eps);
#pragma unroll
        for (int i = 0; i < V; ++i) {
            row[lane + i * WARP] = __float2half(v[i] * rstd * sc[i] + of[i]);
        }
    }
}

// LayerNorm the columns [m0, m0+R) of the fp32 channel-major x_cm [C, M] into the rows of s
// [R][S_LD]. A thread holds one column's slice of C/G channels; red is kThreads floats.
template <int R, int C, int S_LD>
__device__ __forceinline__ void ln_columns(__half* s, float* red, const float* __restrict__ x_cm,
                                           int M, int m0, const float* __restrict__ scale,
                                           const float* __restrict__ offset, float eps) {
    constexpr int G = kThreads / R, V = C / G;
    static_assert(kThreads % R == 0 && C % (2 * G) == 0, "a thread takes channel pairs");
    const int r = threadIdx.x % R, g = threadIdx.x / R, gm = m0 + r;
    float v[V], sum = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] = gm < M ? x_cm[(long long)(g * V + i) * M + gm] : 0.f;
        sum += v[i];
    }
    red[g * R + r] = sum;
    __syncthreads();
    float mean = 0.f;
#pragma unroll
    for (int j = 0; j < G; ++j) {
        mean += red[j * R + r];
    }
    mean *= 1.f / C;
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] -= mean;
        sq += v[i] * v[i];
    }
    __syncthreads();
    red[g * R + r] = sq;
    __syncthreads();
    float var = 0.f;
#pragma unroll
    for (int j = 0; j < G; ++j) {
        var += red[j * R + r];
    }
    const float rstd = rsqrtf(var * (1.f / C) + eps);
#pragma unroll
    for (int i = 0; i < V; i += 2) {
        const int c = g * V + i;
        *reinterpret_cast<__half2*>(s + r * S_LD + c) = __floats2half2_rn(
            v[i] * rstd * scale[c] + offset[c], v[i + 1] * rstd * scale[c + 1] + offset[c + 1]);
    }
}

// bias[h][m0+r] = s[r] . wbT[h] for the R rows of s, a warp per row, staged through hb
// [kMaxHeads][R] so the channel-major store is coalesced.
template <int R, int K, int S_LD>
__device__ __forceinline__ void head_bias(const __half* s, const __half* __restrict__ wbT,
                                          int heads, __half* hb, __half* __restrict__ bias, int M,
                                          int m0) {
    constexpr int V = K / WARP;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    float w[kMaxHeads][V];
#pragma unroll
    for (int h = 0; h < kMaxHeads; ++h) {
#pragma unroll
        for (int i = 0; i < V; ++i) {
            w[h][i] = h < heads ? __half2float(wbT[h * K + lane + i * WARP]) : 0.f;
        }
    }
    for (int r = warp; r < R; r += kThreads / WARP) {
        float v[V];
#pragma unroll
        for (int i = 0; i < V; ++i) {
            v[i] = __half2float(s[r * S_LD + lane + i * WARP]);
        }
#pragma unroll
        for (int h = 0; h < kMaxHeads; ++h) {
            if (h < heads) {
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < V; ++i) {
                    d += v[i] * w[h][i];
                }
                d = warp_sum(d);
                if (lane == 0) {
                    hb[h * R + r] = __float2half(d);
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < heads * R; i += kThreads) {
        const int h = i / R, r = i - h * R;
        if (m0 + r < M) {
            bias[(long long)h * M + m0 + r] = hb[i];
        }
    }
}
