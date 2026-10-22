"""The LayerNorm-fused projections as fused_ops selects them, against colabfold-kernels' float32 references.

  CK_SRC=<colabfold-kernels>/src KERNEL_DIR=<dir with libvolta_*.so> CC=75 python tests/test_fused.py

CC picks the family: 75 takes the mma kernels (libvolta_ops.so), 70 the wmma ones
(libvolta_gdp_wmma.so). Covers ln_gated_dual_proj, trimul_epilogue_ln, ln_projection,
output_projection, transition and the channel-major gated_dual_proj, at row counts 1, 59,
4101 and a 260 x 260 pair, widths 128 and 64, and the MSA's 256 for the transition. Skipping padded row blocks (keep) is in colabfold-kernels' test_padding_skip.py.
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
bad = 0

ROWS = (1, 59, 4101, 260 * 260)
WIDTHS = ((128, 128), (64, 64), (128, 64))   # (pair channels, intermediate / packed slot width)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float("inf")
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))


def f(*sh, s=.5, mean=0.):
  return jnp.asarray(rng.standard_normal(sh) * s + mean, dt)


def f32(*sh, s=.5, mean=0.):
  return jnp.asarray(rng.standard_normal(sh) * s + mean, jnp.float32)


def report(name, err, tol=0.02):
  global bad
  ok = err < tol
  bad += not ok
  print(f"  {'ok ' if ok else 'BAD'} {name:<52} {err:.2e}", flush=True)


def check(name, fn, ref, args, tol=0.02):
  """Forward of every output against the float32 reference on the same (float16) inputs."""
  outs, refs = jax.tree.leaves(fn(*args)), jax.tree.leaves(ref(*args))
  if len(outs) != len(refs):
    report(name + " (result count)", float("inf"))
    return
  report(name, max(rel(o, r) for o, r in zip(outs, refs)), tol)


def calls(fn, args, target):
  """True if fn lowers to the kernel: a dispatcher that silently fell back to XLA would not."""
  return f"@{target}(" in jax.jit(fn).lower(*args).as_text()


def sym(name):
  return name if CC >= 75 else name + "Wmma"


def grad_check(name, fn, ref, args, diff):
  """The gradient goes through the reference; check it reaches every differentiable input."""
  def g(h):
    loss = lambda *d: sum(jnp.sum(o.astype(jnp.float32)) for o in jax.tree.leaves(
        h(*[d[diff.index(i)] if i in diff else a for i, a in enumerate(args)])))
    return jax.grad(loss, argnums=tuple(range(len(diff))))(*[args[i] for i in diff])
  report(name + " grad", max(rel(a, b) for a, b in zip(g(fn), g(ref))), 0.03)


print(f"sm_{CC}: {KD}")
for kernel in ("ln_gated_dual_proj", "trimul_epilogue", "ln_projection", "output_projection", "transition"):
  if not V.fused_available(CC, kernel):
    print(f"  BAD {kernel}: no {sym(V._FUSED_SYMBOLS[kernel])} in {KD}")
    bad += 1

# ---- ln_gated_dual_proj and trimul_epilogue_ln -------------------------------
ep_ln = fused_ops.trimul_epilogue_ln(gc, dt)
for a, ci in WIDTHS:
  gdp = fused_ops.ln_gated_dual_proj(gc, dt, a, ci)
  if gdp is None or ep_ln is None:
    report(f"ln_gdp / epilogue a={a} ci={ci}: no kernel selected", float("inf"))
    continue
  for m in ROWS:
    x = f(m, a, s=2.)
    mask = jnp.asarray(rng.uniform(size=m) > .1, dt)
    sc, of = f32(a, s=.1, mean=1.), f32(a, s=.1)
    wp, wg, bp, bg = f(a, 2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(2 * ci, s=.1)
    args = [x, sc, of, wp, bp, wg, bg, mask]
    check(f"ln_gdp a={a} ci={ci} m={m}", gdp, tm.ref_ln_gdp, args)
    if m == 59:
      report(f"ln_gdp a={a} ci={ci} calls {sym('VoltaLnGdp')}", 0. if calls(gdp, args, sym("VoltaLnGdp")) else 1.)
      grad_check(f"ln_gdp a={a} ci={ci} m={m}", gdp, tm.ref_ln_gdp, args, [0, 1, 3])

    # the triangle product is fp32 and wide; one case sits far from zero for the two-pass statistics
    for amp, mean in ((50., 0.), (5., 1e3)):
      x_cm = f32(ci, m, s=amp, mean=mean)
      ep = [f32(ci, s=.1, mean=1.), f32(ci, s=.1), f(ci, a, s=.1), f(a, s=.1), f(a, a, s=.1), f(a, s=.1)]
      eargs = [x_cm, x, sc, of, *ep]
      for add in (False, True):
        check(f"epilogue_ln a={a} ci={ci} m={m} x_cm~{mean:g}+-{amp:g} add_input={add}",
              lambda *z, add=add: ep_ln(*z, add_input=add),
              lambda *z, add=add: tm.ref_trimul_epilogue_ln(*z, add_input=add), eargs)
    if m == 59:
      report(f"epilogue_ln a={a} ci={ci} calls {sym('VoltaTrimulEpilogue')}",
             0. if calls(lambda *z: ep_ln(*z, add_input=True), eargs, sym("VoltaTrimulEpilogue")) else 1.)
      grad_check(f"epilogue_ln a={a} ci={ci} m={m}", lambda *z: ep_ln(*z, add_input=True),
                 lambda *z: tm.ref_trimul_epilogue_ln(*z, add_input=True), eargs, [0, 1, 2, 6])

# the whole fused triangle multiplication, as AF2's _ln_fused_triangle_multiplication chains it
for a, ci in WIDTHS[:2]:
  gdp = fused_ops.ln_gated_dual_proj(gc, dt, a, ci)
  for n in (59, 260):
    x, mask = f(n * n, a, s=2.), jnp.asarray(rng.uniform(size=n * n) > .1, dt)
    in_ln = [f32(a, s=.1, mean=1.), f32(a, s=.1)]
    proj = [f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1)]
    ep = [f32(ci, s=.1, mean=1.), f32(ci, s=.1), f(ci, a, s=.1), f(a, s=.1), f(a, a, s=.1), f(a, s=.1)]

    def trimul(gdp, epilogue, wide):
      left, right = gdp(x, *in_ln, proj[0], proj[1], proj[2], proj[3], mask)
      tri = jnp.einsum('cik,cjk->cij', left.reshape(ci, n, n), right.reshape(ci, n, n), preferred_element_type=wide)
      return epilogue(tri.reshape(ci, -1), x, *in_ln, *ep)

    got = trimul(gdp, lambda *z: ep_ln(*z, add_input=True), jnp.float32)
    want = trimul(tm.ref_ln_gdp, lambda *z: tm.ref_trimul_epilogue_ln(*z, add_input=True), jnp.float32)
    report(f"ln_gdp -> einsum -> epilogue_ln a={a} ci={ci} n={n}", rel(got, want), 0.03)

# the template stack vmaps the kernels over templates
a, ci, m, t = 64, 64, 900, 3
gdp = fused_ops.ln_gated_dual_proj(gc, dt, a, ci)
xs, masks = f(t, m, a, s=2.), jnp.asarray(rng.uniform(size=(t, m)) > .1, dt)
w = [f32(a, s=.1, mean=1.), f32(a, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1)]
got = jax.vmap(lambda x, mk: gdp(x, *w, mk))(xs, masks)
want = jax.vmap(lambda x, mk: tm.ref_ln_gdp(x, *w, mk))(xs, masks)
report(f"ln_gdp vmapped over {t} templates", max(rel(u, v) for u, v in zip(got, want)))

# without the LayerNorm, gated_dual_proj's channel-major halves stay on the validated VoltaGdp
for act, name in ((None, "sigmoid"), (jax.nn.swish, "swish")):
  g = fused_ops.gated_dual_proj(gc, dt, act)
  for a, ci in WIDTHS:
    m = 4101
    x, mask = f(m, a), jnp.asarray(rng.uniform(size=m) > .1, dt)
    wp, wg, bp, bg = f(a, 2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(2 * ci, s=.1)
    split = lambda *z: g(*z, split=True, channel_major=True)
    def ref(x, wp, bp, wg, bg, mask, act=jax.nn.sigmoid if act is None else act):
      xf = x.astype(jnp.float32)
      y = mask.astype(jnp.float32)[:, None] * (xf @ wp + bp) * act(xf @ wg + bg)
      return y[:, :ci].T, y[:, ci:].T
    args = [x, wp, bp, wg, bg, mask]
    check(f"gdp split channel_major {name} a={a} ci={ci} m={m}", split, ref, args)
    gdp_sym, text = "VoltaGdp" if CC >= 75 else "VoltaGdpWmma", jax.jit(split).lower(*args).as_text()
    report(f"gdp split channel_major {name} a={a} ci={ci} calls {gdp_sym}",
           0. if f"@{gdp_sym}(" in text and "VoltaLnGdp" not in text else 1.)

# ln_gdp equals the old legacy path it replaces: VoltaLayerNorm, then the unsplit VoltaGdp, transposed
ln = fused_ops.layer_norm(gc, dt)
m, a, ci = 4101, 128, 128
x, sc, of = f(m, a, s=2.), f32(a, s=.1, mean=1.), f32(a, s=.1)
mask = jnp.asarray(rng.uniform(size=m) > .1, dt)
wp, wg, bp, bg = f(a, 2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(2 * ci, s=.1)
y = V.volta_gated_dual_proj(ln(x, sc, of), wp, bp, wg, bg, mask, CC)
old_out = y[:, :ci].T, y[:, ci:].T
new_out = fused_ops.ln_gated_dual_proj(gc, dt, a, ci)(x, sc, of, wp, bp, wg, bg, mask)
report("ln_gdp vs layer_norm + gated_dual_proj", max(rel(u, v) for u, v in zip(new_out, old_out)), 1e-2)

# ---- ln_projection and output_projection --------------------------------------
for a, n in WIDTHS:
  lp = fused_ops.ln_projection(gc, dt, a, n)
  bias_only = fused_ops.ln_projection(gc, dt, a)
  if lp is None or bias_only is None:
    report(f"ln_projection a={a} n={n}: no kernel selected", float("inf"))
    continue
  for m in ROWS:
    x, sc, of = f(m, a, s=2.), f32(a, s=.1, mean=1.), f32(a, s=.1)
    for slots, heads in ((4, 4), (3, 8)):
      args = [x, sc, of, f(slots, a, n, s=.1), f(a, heads, s=.1)]
      check(f"ln_projection a={a} {slots}x{n} {heads} heads m={m}", lp, pa.ref_ln_projection, args)
      if m == 59 and slots == 4:
        report(f"ln_projection a={a} n={n} calls {sym('VoltaLnProj')}", 0. if calls(lp, args, sym("VoltaLnProj")) else 1.)
        grad_check(f"ln_projection a={a} n={n} m={m}", lp, pa.ref_ln_projection, args, [0, 1, 3, 4])
    check(f"ln_projection a={a} bias only, 8 heads m={m}", lambda x, s, o, wb: bias_only(x, s, o, None, wb),
          lambda x, s, o, wb: pa.ref_ln_projection(x, s, o, None, wb), [x, sc, of, f(a, 8, s=.1)])
  # more heads than the kernel reduces fall back to XLA, still correct
  x, sc, of = f(59, a, s=2.), f32(a, s=.1, mean=1.), f32(a, s=.1)
  check(f"ln_projection a={a} 16 heads (XLA)", lp, pa.ref_ln_projection, [x, sc, of, f(4, a, n, s=.1), f(a, 16, s=.1)])

for k, c in ((128, 128), (64, 64), (128, 64), (64, 128)):
  op = fused_ops.output_projection(gc, dt, k, c)
  if op is None:
    report(f"output_projection k={k} c={c}: no kernel selected", float("inf"))
    continue
  for m in ROWS:
    o, w, b, r = f(m, k), f(k, c, s=.1), f(c, s=.1), f(m, c, s=2.)
    check(f"output_projection k={k} c={c} m={m}", op, pa.ref_output_projection, [o, w, b, r])
    check(f"output_projection k={k} c={c} m={m} no residual", op, pa.ref_output_projection, [o, w, b])
    if m == 59:
      report(f"output_projection k={k} c={c} calls {sym('VoltaOutProj')}",
             0. if calls(op, [o, w, b, r], sym("VoltaOutProj")) else 1.)
      grad_check(f"output_projection k={k} c={c} m={m}", op, pa.ref_output_projection, [o, w, b, r], [0, 1, 3])

# ---- transition: the pair stack's factor 4, the template stack's 2 ---------------
for c, factor in ((128, 4), (64, 2), (64, 4), (256, 4)):
  t = fused_ops.transition(gc, dt, c, factor * c)
  if t is None:
    report(f"transition c={c} x{factor}: no kernel selected", float("inf"))
    continue
  hd = factor * c
  for m in ROWS:
    # AF2 passes [batch, N, C]
    x = f(1, m, c, s=2.)
    args = [x, f32(c, s=.1, mean=1.), f32(c, s=.1), f(c, hd, s=c ** -.5), f(hd, s=.1), f(hd, c, s=hd ** -.5), f(c, s=.1)]
    for add in (False, True):
      check(f"transition c={c} x{factor} m={m} add_input={add}", lambda *z, add=add: t(*z, add_input=add),
            lambda *z, add=add: tr.ref_transition(*z, add_input=add), args)
    if m == 59:
      report(f"transition c={c} x{factor} calls {sym('VoltaTransition')}",
             0. if calls(lambda *z: t(*z, add_input=True), args, sym("VoltaTransition")) else 1.)
      grad_check(f"transition c={c} x{factor} m={m}", lambda *z: t(*z, add_input=True),
                 lambda *z: tr.ref_transition(*z, add_input=True), args, [0, 1, 3, 5])

# other widths decline to the caller's own path
declined = (fused_ops.ln_gated_dual_proj(gc, dt, 256, 128), fused_ops.ln_projection(gc, dt, 128, 32),
            fused_ops.output_projection(gc, dt, 256, 256), fused_ops.ln_projection(gc, dt, 256, 256),
            fused_ops.ln_projection(gc, jnp.bfloat16, 128, 128),
            fused_ops.transition(gc, dt, 512, 2048), fused_ops.transition(gc, dt, 128, 384))
report("other widths and dtypes decline", 0. if all(d is None for d in declined) else 1.)

print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)
