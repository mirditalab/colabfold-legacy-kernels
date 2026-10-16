// The FFI handlers of the fused units. The including unit defines launch_ln_gdp, launch_epilogue,
// launch_proj and launch_transition for its family, and FUSED_SYM, which names a handler for it.
#pragma once

#include <string>

#include "xla/ffi/api/ffi.h"

#include "volta_gdp.h"

namespace ffi = xla::ffi;
using F16 = ffi::Buffer<ffi::DataType::F16>;
using F32 = ffi::Buffer<ffi::DataType::F32>;

// The widths AF2 uses: pair 128, template stack 64.
static bool fused_width(int64_t c) {
    return c == 64 || c == 128;
}

static ffi::Error launch_status(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string(what) + " launch: " + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// left, right [P/2, M] = the two halves of mask * (y@wp + bp) * sigmoid(y@wg + bg), channel
// major, y = LN(x). The weights come transposed, wpT/wgT [P, K].
static ffi::Error LnGdpImpl(cudaStream_t stream, F16 x, F32 scale, F32 offset, F16 wpT, F16 bp,
                            F16 wgT, F16 bg, F16 mask, ffi::Result<F16> left,
                            ffi::Result<F16> right, float eps) {
    auto dx = x.dimensions(), dw = wpT.dimensions();
    if (dx.size() != 2 || dw.size() != 2 || dw[1] != dx[1]) {
        return ffi::Error::InvalidArgument("ln_gdp: x must be [M,K] and wpT [P,K]");
    }
    const int64_t M = dx[0], K = dx[1], P = dw[0];
    if (!fused_width(K) || P % 2 || !fused_width(P / 2)) {
        return ffi::Error::InvalidArgument("ln_gdp: K and P/2 must be 64 or 128");
    }
    if (wgT.element_count() != (size_t)(P * K) || bp.element_count() != (size_t)P ||
        bg.element_count() != (size_t)P || mask.element_count() != (size_t)M ||
        left->element_count() != (size_t)(P / 2 * M) || scale.element_count() != (size_t)K ||
        offset.element_count() != (size_t)K) {
        return ffi::Error::InvalidArgument("ln_gdp: shapes do not match x [M,K], wpT [P,K]");
    }
    if (M == 0) {
        return ffi::Error::Success();
    }
    auto f16 = [](auto& b) { return reinterpret_cast<const __half*>(b.typed_data()); };
    const LnGdpArgs a{f16(x),
                      f16(wpT),
                      f16(bp),
                      f16(wgT),
                      f16(bg),
                      f16(mask),
                      scale.typed_data(),
                      offset.typed_data(),
                      reinterpret_cast<__half*>(left->typed_data()),
                      reinterpret_cast<__half*>(right->typed_data()),
                      (int)M,
                      (int)P,
                      eps};
    (K == 64 ? launch_ln_gdp<64> : launch_ln_gdp<128>)(stream, a);
    return launch_status("ln_gdp");
}

template <int C>
static void epilogue_dispatch(cudaStream_t stream, const EpilogueArgs& a, int64_t CA, bool add) {
    if (CA == 64) {
        (add ? launch_epilogue<C, 64, true> : launch_epilogue<C, 64, false>)(stream, a);
    } else {
        (add ? launch_epilogue<C, 128, true> : launch_epilogue<C, 128, false>)(stream, a);
    }
}

// out [M,O] = (LN(x_cm.T)@wo + bo) * sigmoid(LN_in(x)@wg + bg), + x with add_input.
static ffi::Error TrimulEpilogueImpl(cudaStream_t stream, F32 x_cm, F16 x, F32 in_scale,
                                     F32 in_offset, F32 ln_scale, F32 ln_offset, F16 woT, F16 bo,
                                     F16 wgT, F16 bg, ffi::Result<F16> out, float eps,
                                     bool add_input) {
    auto dc = x_cm.dimensions(), dx = x.dimensions(), dw = woT.dimensions();
    if (dc.size() != 2 || dx.size() != 2 || dw.size() != 2 || dx[0] != dc[1] || dw[1] != dc[0]) {
        return ffi::Error::InvalidArgument("trimul_epilogue: x_cm [C,M], x [M,CA], woT [O,C]");
    }
    const int64_t C = dc[0], M = dc[1], CA = dx[1], O = dw[0];
    if (!fused_width(C) || !fused_width(CA) || !fused_width(O) || (add_input && O != CA)) {
        return ffi::Error::InvalidArgument(
            "trimul_epilogue: C, CA and O must be 64 or 128, O == CA with add_input");
    }
    if (wgT.element_count() != (size_t)(O * CA) || bo.element_count() != (size_t)O ||
        bg.element_count() != (size_t)O || in_scale.element_count() != (size_t)CA ||
        in_offset.element_count() != (size_t)CA || ln_scale.element_count() != (size_t)C ||
        ln_offset.element_count() != (size_t)C || out->element_count() != (size_t)(M * O)) {
        return ffi::Error::InvalidArgument("trimul_epilogue: shapes do not match");
    }
    if (M == 0) {
        return ffi::Error::Success();
    }
    auto f16 = [](auto& b) { return reinterpret_cast<const __half*>(b.typed_data()); };
    const EpilogueArgs a{x_cm.typed_data(),
                         f16(x),
                         f16(woT),
                         f16(bo),
                         f16(wgT),
                         f16(bg),
                         in_scale.typed_data(),
                         in_offset.typed_data(),
                         ln_scale.typed_data(),
                         ln_offset.typed_data(),
                         reinterpret_cast<__half*>(out->typed_data()),
                         (int)M,
                         (int)O,
                         eps};
    (C == 64 ? epilogue_dispatch<64> : epilogue_dispatch<128>)(stream, a, CA, add_input);
    return launch_status("trimul_epilogue");
}

// p [M,n] = LN(x)@w and bias [heads,M] = (LN(x)@wb).T; n == 0 or heads == 0 leaves that result
// alone, then a one-element buffer.
static ffi::Error LnProjImpl(cudaStream_t stream, F16 x, F32 scale, F32 offset, F16 wT, F16 wbT,
                             ffi::Result<F16> p, ffi::Result<F16> bias, float eps, int64_t n,
                             int64_t heads) {
    auto dx = x.dimensions();
    if (dx.size() != 2) {
        return ffi::Error::InvalidArgument("ln_projection: x must be [M,K]");
    }
    const int64_t M = dx[0], K = dx[1];
    if (!fused_width(K) || n < 0 || n % kTileN || heads < 0 || heads > kMaxHeads) {
        return ffi::Error::InvalidArgument(
            "ln_projection: K must be 64 or 128, n a multiple of 64, at most 8 heads");
    }
    if (scale.element_count() != (size_t)K || offset.element_count() != (size_t)K ||
        (n && (wT.element_count() != (size_t)(n * K) || p->element_count() != (size_t)(M * n))) ||
        (heads && (wbT.element_count() != (size_t)(heads * K) ||
                   bias->element_count() != (size_t)(heads * M)))) {
        return ffi::Error::InvalidArgument("ln_projection: shapes do not match");
    }
    if (M == 0) {
        return ffi::Error::Success();
    }
    auto f16 = [](auto& b) { return reinterpret_cast<const __half*>(b.typed_data()); };
    const ProjArgs a{f16(x),
                     n ? f16(wT) : nullptr,
                     nullptr,
                     nullptr,
                     heads ? f16(wbT) : nullptr,
                     scale.typed_data(),
                     offset.typed_data(),
                     n ? reinterpret_cast<__half*>(p->typed_data()) : nullptr,
                     heads ? reinterpret_cast<__half*>(bias->typed_data()) : nullptr,
                     (int)M,
                     (int)n,
                     (int)heads,
                     eps};
    (K == 64 ? launch_proj<64, true> : launch_proj<128, true>)(stream, a);
    return launch_status("ln_projection");
}

// out [M,N] = o@w + b, + residual with has_residual (else a one-element buffer).
static ffi::Error OutProjImpl(cudaStream_t stream, F16 o, F16 wT, F16 b, F16 residual,
                              ffi::Result<F16> out, bool has_residual) {
    auto dx = o.dimensions(), dw = wT.dimensions();
    if (dx.size() != 2 || dw.size() != 2 || dw[1] != dx[1]) {
        return ffi::Error::InvalidArgument("output_projection: o must be [M,K] and wT [N,K]");
    }
    const int64_t M = dx[0], K = dx[1], N = dw[0];
    if (!fused_width(K) || !fused_width(N)) {
        return ffi::Error::InvalidArgument("output_projection: K and N must be 64 or 128");
    }
    if (b.element_count() != (size_t)N || out->element_count() != (size_t)(M * N) ||
        (has_residual && residual.element_count() != (size_t)(M * N))) {
        return ffi::Error::InvalidArgument("output_projection: shapes do not match");
    }
    if (M == 0) {
        return ffi::Error::Success();
    }
    auto f16 = [](auto& t) { return reinterpret_cast<const __half*>(t.typed_data()); };
    const ProjArgs a{f16(o),  f16(wT), f16(b),  has_residual ? f16(residual) : nullptr,
                     nullptr, nullptr, nullptr, reinterpret_cast<__half*>(out->typed_data()),
                     nullptr, (int)M,  (int)N,  0,
                     0.f};
    (K == 64 ? launch_proj<64, false> : launch_proj<128, false>)(stream, a);
    return launch_status("output_projection");
}

// out [M,C] = relu(LN(x)@w1 + b1)@w2 + b2, + x with add_input. w2T holds w2 [H,C] as kHidden-row
// chunks, each transposed: [H/kHidden][C][kHidden].
static ffi::Error TransitionImpl(cudaStream_t stream, F16 x, F32 scale, F32 offset, F16 w1T, F16 b1,
                                 F16 w2T, F16 b2, ffi::Result<F16> out, float eps, bool add_input) {
    auto dx = x.dimensions(), dw = w1T.dimensions();
    if (dx.size() != 2 || dw.size() != 2 || dw[1] != dx[1]) {
        return ffi::Error::InvalidArgument("transition: x must be [M,C] and w1T [H,C]");
    }
    const int64_t M = dx[0], C = dx[1], H = dw[0];
    if (!fused_width(C) || (H != 2 * C && H != 4 * C)) {
        return ffi::Error::InvalidArgument("transition: C must be 64 or 128, H 2C or 4C");
    }
    if (scale.element_count() != (size_t)C || offset.element_count() != (size_t)C ||
        b1.element_count() != (size_t)H || w2T.element_count() != (size_t)(H * C) ||
        b2.element_count() != (size_t)C || out->element_count() != (size_t)(M * C)) {
        return ffi::Error::InvalidArgument("transition: shapes do not match");
    }
    if (M == 0) {
        return ffi::Error::Success();
    }
    auto f16 = [](auto& b) { return reinterpret_cast<const __half*>(b.typed_data()); };
    const TransitionArgs a{f16(x),
                           f16(w1T),
                           f16(b1),
                           f16(w2T),
                           f16(b2),
                           scale.typed_data(),
                           offset.typed_data(),
                           reinterpret_cast<__half*>(out->typed_data()),
                           (int)M,
                           (int)H,
                           eps};
    if (C == 64) {
        (add_input ? launch_transition<64, true> : launch_transition<64, false>)(stream, a);
    } else {
        (add_input ? launch_transition<128, true> : launch_transition<128, false>)(stream, a);
    }
    return launch_status("transition");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FUSED_SYM(VoltaLnGdp), LnGdpImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<F16>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Ret<F16>()
                                  .Ret<F16>()
                                  .Attr<float>("eps"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(FUSED_SYM(VoltaTrimulEpilogue), TrimulEpilogueImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<F32>()
                                  .Arg<F16>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Ret<F16>()
                                  .Attr<float>("eps")
                                  .Attr<bool>("add_input"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(FUSED_SYM(VoltaLnProj), LnProjImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<F16>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Ret<F16>()
                                  .Ret<F16>()
                                  .Attr<float>("eps")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("heads"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(FUSED_SYM(VoltaOutProj), OutProjImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Ret<F16>()
                                  .Attr<bool>("has_residual"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(FUSED_SYM(VoltaTransition), TransitionImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<F16>()
                                  .Arg<F32>()
                                  .Arg<F32>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Arg<F16>()
                                  .Ret<F16>()
                                  .Attr<float>("eps")
                                  .Attr<bool>("add_input"),
                              {ffi::Traits::kCmdBufferCompatible});
