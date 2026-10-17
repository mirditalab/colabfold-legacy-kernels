// Gate activations both gated dual projection units share.
#pragma once

#include <cuda_fp16.h>

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
