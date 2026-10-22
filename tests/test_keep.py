"""The fused kernels skipping a padded pair's or MSA's row blocks (keep), as colabfold-kernels' Pallas kernels do.

  CK_SRC=<colabfold-kernels>/src KERNEL_DIR=<dir with libvolta_*.so> CC=75 python tests/test_keep.py

With keep, real rows are bit-identical to the call without it, padded rows either that or what a skipped block writes
(zeros, or the residual unchanged where the output aliases it), all finite. A keep below zero past the live block: those
blocks are not written. ln_gated_dual_proj takes its mask as the keep.
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
from colabfold_kernels import fused_ops, volta as V
from colabfold_kernels import packed_attn as pa, transition as tr, tri_mul as tm

KD = os.environ["KERNEL_DIR"]
CC = int(os.environ.get("CC", "75"))
V._package_path = lambda kernel, cc: os.path.join(KD, clk._resolve(kernel, cc)[1])
gc = {"use_pallas": True, "kernel_backend": "cuda_legacy", "compute_capability": CC}
dt = jnp.float16
rng = np.random.default_rng(0)
fails = []


def f(*shape, s=1.0):
  return jnp.asarray(rng.normal(size=shape) * s, dt)


def f32(*shape, s=1.0, mean=0.):
  return jnp.asarray(rng.normal(size=shape) * s + mean, jnp.float32)


def check_keep(name, with_keep, without, keep, pad_value):
  ok, skipped = True, 0.
  for o, b, pv in zip(with_keep, without, pad_value):
    o, b = np.asarray(o.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))
    pv = np.broadcast_to(np.asarray(jnp.asarray(pv).astype(jnp.float32)), o.shape)
    k = np.asarray(keep).astype(bool).reshape((-1,) + (1,) * (o.ndim - 1)) if o.shape[0] == keep.size else \
        np.asarray(keep).astype(bool).reshape((1, -1) + (1,) * (o.ndim - 2))
    k = np.broadcast_to(k, o.shape)
    real = float(np.abs(np.where(k, o - b, 0)).max())
    padded = np.where(k, True, (o == b) | (o == pv))
    ok &= real == 0 and bool(padded.all()) and bool(np.isfinite(o).all())
    skipped = float(np.mean(np.where(k, False, (o == pv) & (o != b))) / max(np.mean(~k), 1e-9))
  print(f"{'ok  ' if ok else 'BAD '}{name}: real rows identical {ok}, padded rows skipped {skipped:.0%}", flush=True)
  if not ok:
    fails.append(name)


print(f"sm_{CC}: {KD}")
zero = jnp.zeros(())
# a pair at a padded length, and an MSA at its full depth (live rows lead), as AF2 passes their masks
cases = [(f"pair S={S} n={n}", S * S, ((jnp.arange(S) < n)[:, None] & (jnp.arange(S) < n)[None, :]).reshape(-1))
         for S, n in [(256, 256), (256, 200), (256, 129), (192, 65)]]
cases += [(f"msa X={X} Y={Y} live={live}", X * Y, jnp.broadcast_to((jnp.arange(X) < live)[:, None], (X, Y)).reshape(-1))
          for X, Y, live in [(508, 200, 1), (508, 200, 37), (64, 130, 64)]]
for case, m, live in cases:
  keep = live.astype(dt)
  for c, n, h in ((128, 128, 4), (256, 256, 8)) if case.startswith("msa") else ((128, 128, 4), (64, 64, 4)):
    x = f(m, c, s=2.)
    sc, of = f32(c, s=.1, mean=1.), f32(c, s=.1)
    t = fused_ops.transition(gc, dt, c, 4 * c)
    tw = (sc, of, f(c, 4 * c, s=c ** -.5), f(4 * c, s=.1), f(4 * c, c, s=(4 * c) ** -.5), f(c, s=.1))
    for add in (False, True):
      check_keep(f"transition {c} add_input={add} {case}", [t(x, *tw, keep, add_input=add)],
                 [t(x, *tw, add_input=add)], keep, [x if add else zero])
    if c > 128:  # the MSA's 256-channel projections run in XLA
      continue
    lp = fused_ops.ln_projection(gc, dt, c, n)
    w, wb = f(4, c, n, s=.1), f(c, h, s=.1)
    check_keep(f"ln_projection {c}->4x{n} {case}", lp(x, sc, of, w, wb, keep), lp(x, sc, of, w, wb), keep,
               [zero, zero])
    op = fused_ops.output_projection(gc, dt, n, c)
    o, wo, bo = f(m, n), f(n, c, s=.1), f(c, s=.1)
    check_keep(f"output_projection {n}->{c} {case}", [op(o, wo, bo, x, keep)], [op(o, wo, bo, x)], keep, [x])
    check_keep(f"output_projection no residual {n}->{c} {case}", [op(o, wo, bo, None, keep)], [op(o, wo, bo)],
               keep, [zero])
    if case.startswith("pair"):
      ep = fused_ops.trimul_epilogue_ln(gc, dt)
      tri = f32(c, m, s=20.)
      g = (sc, of, f32(c, s=.1, mean=1.), f32(c, s=.1), f(c, c, s=.1), f(c, s=.1), f(c, c, s=.1), f(c, s=.1))
      for add in (False, True):
        check_keep(f"trimul_epilogue_ln {c} add_input={add} {case}", [ep(tri, x, *g, keep, add_input=add)],
                   [ep(tri, x, *g, add_input=add)], keep, [x if add else zero])
      gdp = fused_ops.ln_gated_dual_proj(gc, dt, c, c)
      gw = (sc, of, f(c, 2 * c, s=.1), f(2 * c, s=.1), f(c, 2 * c, s=.1), f(2 * c, s=.1))
      out, ref = gdp(x, *gw, keep), tm.ref_ln_gdp(x, *gw, keep)
      ok = all(bool(np.isfinite(np.asarray(o, np.float32)).all())
               and float(np.abs(np.asarray(o, np.float32) - np.asarray(r)).max()) / (float(np.abs(np.asarray(r)).max())
                                                                                     + 1e-6) < 0.02
               and float(np.abs(np.asarray(o, np.float32)[:, ~np.asarray(live)]).max(initial=0)) == 0
               for o, r in zip(out, ref))
      print(f"{'ok  ' if ok else 'BAD '}ln_gdp {c} {case}: as the reference, padded rows exactly zero", flush=True)
      if not ok:
        fails.append(f"ln_gdp {c} {case}")

# keep below zero past the live block rounded up (what a trimmed attention or the live block's contraction reads):
# ln_projection and ln_gated_dual_proj write nothing there; real rows as without keep, padding inside the block as
# without keep or zeros
for S, n in [(256, 200), (384, 129), (192, 65)]:
  R = -(-n // 128) * 128
  tok, ins = jnp.arange(S) < n, jnp.arange(S) < R
  mask = (tok[:, None] & tok[None, :]).astype(dt)
  region = jnp.where(ins[:, None] & ins[None, :], mask, -1).astype(dt)
  inside = np.asarray(ins[:, None] & ins[None, :]).reshape(-1)
  c = 128
  x = f(S * S, c, s=2.)
  sc, of, w, wb = f32(c, s=.1, mean=1.), f32(c, s=.1), f(4, c, 128, s=.1), f(c, 4, s=.1)
  lp = fused_ops.ln_projection(gc, dt, c, 128)
  out = [np.asarray(t.astype(jnp.float32)) for t in lp(x, sc, of, w, wb, region.reshape(-1))]
  base = [np.asarray(t.astype(jnp.float32)) for t in lp(x, sc, of, w, wb)]
  live = np.asarray(mask).reshape(-1) > 0
  ok = all(np.array_equal(o[live] if o.shape[0] == live.size else o[:, live],
                          b[live] if b.shape[0] == live.size else b[:, live]) for o, b in zip(out, base))
  pad = inside & ~live
  rows = lambda t: t[pad] if t.shape[0] == live.size else t[:, pad]
  ok &= all(np.all((rows(o) == rows(b)) | (rows(o) == 0)) for o, b in zip(out, base))
  gdp = fused_ops.ln_gated_dual_proj(gc, dt, c, c)
  g = (f32(c, s=.1, mean=1.), f32(c, s=.1), f(c, 2 * c, s=.1), f(2 * c, s=.1), f(c, 2 * c, s=.1), f(2 * c, s=.1))
  out = gdp(x, *g, region.reshape(-1))
  ref = tm.ref_ln_gdp(x, *g, mask.reshape(-1))
  for o, r in zip(out, ref):
    o = np.asarray(o.astype(jnp.float32)).reshape(c, S, S)[:, :R, :R]
    r = np.asarray(r).reshape(c, S, S)[:, :R, :R]
    ok &= bool(np.isfinite(o).all()) and float(np.abs(o - r).max() / (np.abs(r).max() + 1e-6)) < 0.02
    # padding by the mask: a live pair's reference can round to exactly zero too
    ok &= float(np.abs(o[:, ~np.asarray(mask[:R, :R] > 0)]).max(initial=0)) == 0
  print(f"{'ok  ' if ok else 'BAD '}keep below zero S={S} n={n}: real rows as without, padding inside the block too or zeros",
        flush=True)
  if not ok:
    fails.append(f"keep below zero S={S} n={n}")

print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(bool(fails))
