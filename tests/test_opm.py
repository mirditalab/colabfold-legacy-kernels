"""The outer product mean kernels as fused_ops selects them, against colabfold-kernels' float32 references.

  CK_SRC=<colabfold-kernels>/src KERNEL_DIR=<dir with libvolta_*.so> CC=75 python tests/test_opm.py

CC picks the family: 75 takes the mma kernels (libvolta_ops.so), 70 the wmma ones (libvolta_gdp_wmma.so). opm_update
at B 1, 37 and 260, D 4, 64 and 260, K 32, 64 and 512, F 128 and 64, with and without the residual; opm_outer at the
same B and D on 16 rows with AF2's 32 channels, against ref_opm_outer (XLA's two f16 GEMMs) and against its rounding
in float32.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
sys.path.insert(0, os.environ.get("CK_SRC", os.path.join(here, "..", "..", "colabfold-kernels", "src")))
import colabfold_legacy_kernels as clk
from colabfold_kernels import fused_ops, opm, volta as V

KD = os.environ["KERNEL_DIR"]
CC = int(os.environ.get("CC", "75"))
V._package_path = lambda kernel, cc: os.path.join(KD, clk._resolve(kernel, cc)[1])
jax.config.update("jax_default_matmul_precision", "highest")   # the float32 references in full precision
gc = {"use_pallas": True, "kernel_backend": "cuda_legacy", "compute_capability": CC}
dt = jnp.float16
rng = np.random.default_rng(0)
bad = 0

BS, DS = (1, 37, 260), (4, 64, 260)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float("inf")
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))


def f(*sh, s=1.):
  return jnp.asarray(rng.standard_normal(sh) * s, dt)


def report(name, err, tol=2e-3):
  global bad
  ok = err < tol
  bad += not ok
  print(f"  {'ok ' if ok else 'BAD'} {name:<56} {err:.2e}", flush=True)


def calls(fn, args, target):
  """True if fn lowers to the kernel: a dispatcher that silently fell back to XLA would not."""
  return f"@{target}(" in jax.jit(fn).lower(*args).as_text()


def sym(name):
  return name if CC >= 75 else name + "Wmma"


def grad_check(name, fn, ref, args, diff):
  """The gradient goes through the reference; check it reaches every differentiable input."""
  def g(h):
    loss = lambda *d: jnp.sum(
        h(*[d[diff.index(i)] if i in diff else a for i, a in enumerate(args)]).astype(jnp.float32))
    return jax.grad(loss, argnums=tuple(range(len(diff))))(*[args[i] for i in diff])
  report(name + " grad", max(rel(a, b) for a, b in zip(g(fn), g(ref))), 0.03)


def outer_f32(left, right, w):
  """ref_opm_outer's rounding in float32: the outer product rounded to f16, the second contraction not."""
  act = jnp.einsum('abc,ade->bced', left.astype(jnp.float32), right.astype(jnp.float32)).astype(dt)
  act = act.reshape(act.shape[0], -1, act.shape[-1]).astype(jnp.float32)
  return jnp.einsum('bkd,kf->bdf', act, w.astype(jnp.float32))


print(f"sm_{CC}: {KD}")
for kernel in ("opm_update", "opm_outer"):
  if not V.fused_available(CC, kernel):
    print(f"  BAD {kernel}: no {sym(V._FUSED_SYMBOLS[kernel])} in {KD}")
    bad += 1

# ---- opm_update: lhs [B, rows*32] against t [rows*32, D, F], AF2's mask counts as the norm -----------------
for F in (128, 64):
  for K in (32, 64, 512):
    up = fused_ops.opm_update(gc, dt, K, F)
    if up is None:
      report(f"opm_update K={K} F={F}: no kernel selected", float("inf"))
      continue
    for B in BS:
      for D in DS:
        lhs, t, bias = f(B, K), f(K, D, F, s=K ** -.5), f(F, s=.1)
        norm = jnp.asarray(rng.integers(1, K // 32 + 1, size=(B, D)), jnp.float32)
        res = f(B, D, F, s=2.)
        for args, tag in (([lhs, t, bias, norm], ""), ([lhs, t, bias, norm, res], " residual")):
          report(f"opm_update B={B} D={D} K={K} F={F}{tag}", rel(up(*args), opm.ref_opm_update(*args)))
        if (B, D) == (37, 64):
          report(f"opm_update K={K} F={F} calls {sym('VoltaOpmUpdate')}",
                 0. if calls(up, [lhs, t, bias, norm, res], sym("VoltaOpmUpdate")) else 1.)
          if K == 64:
            grad_check(f"opm_update K={K} F={F}", up, opm.ref_opm_update, [lhs, t, bias, norm, res], [0, 1, 2, 4])

# the residual is aliased to the output inside the call, never the caller's array
up = fused_ops.opm_update(gc, dt, 64, 128)
lhs, t, bias, res = f(37, 64), f(64, 64, 128, s=.125), f(128, s=.1), f(37, 64, 128)
norm = jnp.ones((37, 64), jnp.float32)
before = np.asarray(res)
jax.block_until_ready(jax.jit(up)(lhs, t, bias, norm, res))
report("opm_update leaves the caller's residual", 0. if (np.asarray(res) == before).all() else 1.)

# ---- opm_outer: AF2's own order on the leading 16 rows ---------------------------------------------------------
for F in (128, 64):
  for B in BS:
    for D in DS:
      outer = fused_ops.opm_outer(gc, dt, 16, 32, D, F)
      if outer is None:
        report(f"opm_outer B={B} D={D} F={F}: no kernel selected", float("inf"))
        continue
      args = [f(16, B, 32), f(16, D, 32), f(32 * 32, F, s=1 / 32)]
      got = outer(*args)
      report(f"opm_outer B={B} D={D} F={F} vs ref_opm_outer", rel(got, opm.ref_opm_outer(*args)))
      report(f"opm_outer B={B} D={D} F={F} vs its rounding in f32", rel(got, outer_f32(*args)))
      if (B, D) == (37, 64):
        report(f"opm_outer F={F} calls {sym('VoltaOpmOuter')}", 0. if calls(outer, args, sym("VoltaOpmOuter")) else 1.)
        grad_check(f"opm_outer F={F}", outer, opm.ref_opm_outer, args, [0, 1, 2])

# leading live rows of a full-depth MSA: the padded rows are zero, so the shortcut equals the full contraction
outer = fused_ops.opm_outer(gc, dt, 16, 32, 64, 128)
left, right, w = f(16, 64, 32), f(16, 64, 32), f(32 * 32, 128, s=1 / 32)
pad = lambda x: jnp.pad(x, ((0, 112), (0, 0), (0, 0)))
report("opm_outer on 16 live rows vs ref_opm_outer on 128",
       rel(outer(left, right, w), opm.ref_opm_outer(pad(left), pad(right), w)))

# shapes the kernels do not take decline to the caller's own path
declined = (fused_ops.opm_update(gc, dt, 48, 128), fused_ops.opm_update(gc, dt, 64, 256),
            fused_ops.opm_update(gc, jnp.bfloat16, 64, 128), fused_ops.opm_outer(gc, dt, 32, 32, 64, 128),
            fused_ops.opm_outer(gc, dt, 16, 16, 64, 128), fused_ops.opm_outer(gc, dt, 16, 32, 66, 128),
            fused_ops.opm_outer(gc, dt, 16, 32, 64, 96))
report("other shapes and dtypes decline", 0. if all(d is None for d in declined) else 1.)

print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)
