"""VoltaMma / VoltaMmaBwd against JAX's own autodiff, and the wmma pair.

  KERNEL_DIR=<dir with libvolta_*.so> python tests/test_attn_bwd.py [mma|wmma]

Neither family needs the card it targets: the mma kernels run on any
Turing-or-newer card and the wmma ones on anything from Volta up, so both can
be developed on an Ampere box and only confirmed on the T4 or V100 they are
meant for.
"""
import ctypes
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

KERNEL_DIR = os.environ.get('KERNEL_DIR', '.')
NEG = -1.0e4

# Every (head dim, block_q, block_k) the forward dispatches, per family.
FWD_CONFIGS = {
    'mma': ((8, 64, 64), (8, 64, 32), (8, 32, 32), (8, 128, 64), (8, 32, 64),
            (32, 64, 64), (32, 64, 32), (32, 32, 64), (32, 32, 32),
            (32, 128, 64), (32, 16, 64), (16, 64, 64), (16, 32, 32),
            (16, 64, 32), (64, 64, 64), (64, 32, 32)),
    'wmma': ((32, 64, 64), (32, 64, 32), (32, 32, 64), (32, 32, 32),
             (32, 128, 64), (32, 128, 32), (16, 64, 64), (16, 32, 32),
             (16, 64, 32), (64, 64, 64), (64, 32, 32), (64, 64, 32),
             (8, 64, 64), (8, 64, 32), (8, 32, 32)),
}

FAMILY = 'mma'   # or 'wmma', set from argv


def register():
  tag = 'mma' if FAMILY == 'mma' else 'wmma'
  name = 'Mma' if FAMILY == 'mma' else 'Wmma'
  for lib, sym in ((f'libvolta_{tag}.so', f'Volta{name}'),
                   (f'libvolta_{tag}_bwd.so', f'Volta{name}Bwd')):
    handle = ctypes.cdll.LoadLibrary(os.path.join(KERNEL_DIR, lib))
    jax.ffi.register_ffi_target(
        sym, jax.ffi.pycapsule(getattr(handle, sym)), platform='CUDA')


def fwd(q, k, v, bias, kmask, scale, bq=64, bk=32, want_lse=True):
  n, h, sq, d = q.shape
  # want_lse false leaves the buffer untouched, so one element is enough
  lse_shape = (n, h, sq) if want_lse else (1,)
  return jax.ffi.ffi_call(
      'VoltaMma' if FAMILY == 'mma' else 'VoltaWmma',
      (jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
       jax.ShapeDtypeStruct(lse_shape, jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, scale=np.float32(scale),
          block_q=np.int64(bq), block_k=np.int64(bk), want_lse=want_lse)


def bwd(q, k, v, bias, kmask, out, dout, lse, scale, want_dbias=True):
  n, h, sq, d = q.shape
  sk = k.shape[2]
  return jax.ffi.ffi_call(
      'VoltaMmaBwd' if FAMILY == 'mma' else 'VoltaWmmaBwd',
      (jax.ShapeDtypeStruct(q.shape, jnp.float16),
       jax.ShapeDtypeStruct(k.shape, jnp.float16),
       jax.ShapeDtypeStruct(v.shape, jnp.float16),
       jax.ShapeDtypeStruct((h, sq, sk), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, out, dout, lse, scale=np.float32(scale),
          want_dbias=want_dbias)


def reference(q, k, v, bias, kmask, scale):
  """fp32 attention with the kernel's own masking convention."""
  logits = scale * jnp.einsum('nhqd,nhkd->nhqk', q, k) + bias[None]
  logits = jnp.where(kmask[:, None, None, :] != 0, logits, NEG)
  p = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('nhqk,nhkd->nhqd', p, v)


def reference_grads(q, k, v, bias, kmask, dout, scale):
  """The same attention and its gradients in float64 numpy. Closed form, so a
  sweep over shapes costs no XLA compile; vjp_case keeps the autodiff check."""
  q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
  bias, dout = np.asarray(bias, np.float64), np.asarray(dout, np.float64)
  m = np.asarray(kmask)[:, None, None, :] != 0
  logits = np.where(m, scale * np.einsum('nhqd,nhkd->nhqk', q, k) + bias[None], NEG)
  mx = logits.max(-1, keepdims=True)
  e = np.exp(logits - mx)
  l = e.sum(-1, keepdims=True)
  p = e / l
  out = np.einsum('nhqk,nhkd->nhqd', p, v)
  lse2 = ((np.log(l) + mx)[..., 0]) * 1.4426950408889634
  dv = np.einsum('nhqk,nhqd->nhkd', p, dout)
  dp = np.einsum('nhqd,nhkd->nhqk', dout, v)
  ds = p * (dp - (p * dp).sum(-1, keepdims=True))
  ds = np.where(m, ds, 0.0)          # a masked logit is a constant
  dq = np.einsum('nhqk,nhkd->nhqd', ds, k) * scale
  dk = np.einsum('nhqk,nhqd->nhkd', ds, q) * scale
  return out, lse2, dq, dk, dv, ds.sum(0)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float('inf')          # a NaN must never max() away behind a number
  # With one key the softmax is 1 whatever the logit is, so the gradient is
  # exactly zero, and zero has no relative scale.
  denom = max(np.abs(b).max(), 1e-3)
  return float(np.abs(a - b).max() / denom)


def inputs(n, h, sq, sk, d, seed, masked_rows=0.15, qk_amp=0.5, vo_amp=0.5):
  # numpy, not jax.random: every distinct shape would otherwise cost an XLA
  # compile, which dominates a sweep over shapes.
  rng = np.random.default_rng(seed)
  f16 = lambda x: jnp.asarray(x, jnp.float16)
  q = f16(rng.standard_normal((n, h, sq, d)) * qk_amp)
  k = f16(rng.standard_normal((n, h, sk, d)) * qk_amp)
  v = f16(rng.standard_normal((n, h, sk, d)) * vo_amp)
  bias = f16(rng.standard_normal((h, sq, sk)) * qk_amp)
  kmask = jnp.asarray(rng.random((n, sk)) > masked_rows, jnp.uint8)
  dout = f16(rng.standard_normal((n, h, sq, d)) * vo_amp)
  return q, k, v, bias, kmask, dout


def run(n=3, h=4, sq=96, sk=96, d=32, seed=0, bq=64, bk=32, kmask=None,
        quiet=False, qk_amp=0.5, vo_amp=0.5):
  q, k, v, bias, km, dout = inputs(n, h, sq, sk, d, seed, qk_amp=qk_amp,
                                   vo_amp=vo_amp)
  if kmask is not None:
    km = kmask
  scale = float(d) ** -0.5

  # lse comes back in the log2 domain, which is what the kernel's exp2 wants.
  ref_out, ref_lse2, ref_dq, ref_dk, ref_dv, ref_dbias = reference_grads(
      q, k, v, bias, km, dout, scale)

  out, lse = fwd(q, k, v, bias, km, scale, bq, bk)
  dq, dk, dv, dbias = bwd(q, k, v, bias, km, out, dout, lse, scale)

  rows = [('out', out, ref_out), ('lse2', lse, ref_lse2), ('dq', dq, ref_dq),
          ('dk', dk, ref_dk), ('dv', dv, ref_dv), ('dbias', dbias, ref_dbias)]
  worst = 0.0
  for name, got, want in rows:
    r = rel(got, want)
    worst = r if worst != worst or r > worst else worst
    if not quiet:
      print(f'  {name:6s} rel {r:.2e}   '
            f'max|ref| {np.abs(np.asarray(want)).max():.3f}')
  return worst


def try_run(**kwargs):
  """None when the card cannot fit the tile: the wmma kernels are sized for
  Volta's 96 KB and a Turing card has 64 KB."""
  try:
    return run(**kwargs)
  except Exception as exc:                   # noqa: BLE001 - only the one case
    if 'KB shared, device allows' in str(exc):
      return None
    raise


def check(label, worst, tol=0.05):
  if worst is None:
    print(f'  -> skip ({label} does not fit this card)')
    return 0
  ok = worst < tol
  print('  ->', 'OK' if ok else 'FAIL', f'({label} worst {worst:.2e})')
  return not ok


def masked_rows_case():
  """A batch element with every key masked: the forward softmax is uniform."""
  print('one batch element with every key masked')
  km = np.ones((3, 96), np.uint8)
  km[1, :] = 0
  worst = try_run(kmask=jnp.asarray(km))
  return check('all-masked', worst)


def coverage_case():
  """Whatever block shape the forward takes, the backward must differentiate:
  it picks its own tiling from the head dim."""
  print('every forward (D, block_q, block_k)')
  bad = 0
  for d, bq, bk in FWD_CONFIGS[FAMILY]:
    try:
      worst = try_run(n=2, h=2, sq=130, sk=130, d=d, bq=bq, bk=bk, quiet=True)
    except Exception as exc:                 # noqa: BLE001 - report, keep going
      print(f'  D={d:3d} bq={bq:3d} bk={bk:3d}  {type(exc).__name__}: '
            f'{str(exc).splitlines()[0][:60]}')
      bad += 1
      continue
    if worst is None:
      print(f'  D={d:3d} bq={bq:3d} bk={bk:3d}  skip, too big for this card')
      continue
    flag = '' if worst < 0.05 else '  FAIL'
    print(f'  D={d:3d} bq={bq:3d} bk={bk:3d}  worst {worst:.2e}{flag}')
    bad += worst >= 0.05
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def amplitude_case():
  """Activations near the fp16 ceiling: dS must not overflow the half it is
  stored in, and the clamp must stay below the logits it hides."""
  print('large activations')
  bad = 0
  for qk, vo in ((0.5, 32.0), (0.5, 90.0), (16.0, 0.5), (32.0, 0.5)):
    worst = try_run(n=2, h=2, d=64, bk=64, qk_amp=qk, vo_amp=vo, quiet=True)
    if worst is None:
      print(f'  qk x{qk:5.1f}  v/dO x{vo:5.1f}   skip, too big for this card')
      continue
    flag = '' if worst < 0.05 else '  FAIL'
    print(f'  qk x{qk:5.1f}  v/dO x{vo:5.1f}   worst {worst:.2e}{flag}')
    bad += worst >= 0.05
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def tiny_case():
  """Shapes below one block, including the Sq=1 template pointwise signature."""
  print('shapes smaller than a block')
  bad = 0
  for sq, sk, d in ((1, 1, 16), (1, 4, 16), (7, 13, 16), (33, 1, 32), (3, 96, 8)):
    worst = try_run(n=2, h=2, sq=sq, sk=sk, d=d, quiet=True)
    if worst is None:
      print(f'  sq={sq:4d} sk={sk:4d} D={d:3d}   skip, too big for this card')
      continue
    flag = '' if worst < 0.05 else '  FAIL'
    print(f'  sq={sq:4d} sk={sk:4d} D={d:3d}   worst {worst:.2e}{flag}')
    bad += worst >= 0.05
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def repeat_case():
  """dK/dV are deterministic. dQ and dBias arrive by atomic add, from every key
  tile and every batch row, so their summation order is not fixed."""
  print('repeated calls')
  q, k, v, bias, km, dout = inputs(3, 4, 96, 96, 32, 0)
  scale = 32.0 ** -0.5
  out, lse = fwd(q, k, v, bias, km, scale)
  a = bwd(q, k, v, bias, km, out, dout, lse, scale)
  b = bwd(q, k, v, bias, km, out, dout, lse, scale)
  bad = 0
  for name, x, y in zip(('dk', 'dv'), a[1:3], b[1:3]):
    if not np.array_equal(np.asarray(x), np.asarray(y)):
      print(f'  {name} differs between identical calls')
      bad += 1
  for name, i in (('dq', 0), ('dbias', 3)):
    r = rel(a[i], b[i])
    print(f'  {name:5s} run-to-run rel {r:.2e} (atomics, so only bounded)')
    bad += r >= 1e-3
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def vjp_case():
  """The kernels wired as a custom_vjp, against jax.grad of the reference."""
  print('as a custom_vjp')
  q, k, v, bias, km, dout = inputs(2, 4, 96, 96, 32, 0)
  scale = 32.0 ** -0.5

  @jax.custom_vjp
  def fused(q, k, v, bias):
    return fwd(q, k, v, bias, km, scale)[0]

  def fused_fwd(q, k, v, bias):
    out, lse = fwd(q, k, v, bias, km, scale)
    return out, (q, k, v, bias, out, lse)

  def fused_bwd(res, dout):
    q, k, v, bias, out, lse = res
    dq, dk, dv, dbias = bwd(q, k, v, bias, km, out, dout, lse, scale)
    return dq, dk, dv, dbias.astype(bias.dtype)

  fused.defvjp(fused_fwd, fused_bwd)
  loss = lambda f, *a: jnp.sum(f(*a).astype(jnp.float32) * dout.astype(jnp.float32))
  got = jax.grad(loss, argnums=(1, 2, 3, 4))(fused, q, k, v, bias)
  ref_f = lambda a, b, c, e: reference(a, b, c, e, km, scale)
  want = jax.grad(loss, argnums=(1, 2, 3, 4))(
      ref_f, *[x.astype(jnp.float32) for x in (q, k, v, bias)])
  bad = 0
  for name, g, r in zip(('dq', 'dk', 'dv', 'dbias'), got, want):
    rr = rel(g, r)
    bad += rr >= 0.05
    print(f'  {name:6s} rel {rr:.2e}')
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def xla_case():
  """Against the path that runs when the kernels are off. float64 is truth, so
  this says whether the kernel is at least as close to it as XLA in half is."""
  print('against the unfused XLA path')
  bad = 0
  for n, h, sq, sk, d in ((2, 4, 96, 96, 32), (2, 4, 70, 83, 16), (1, 8, 128, 128, 64)):
    q, k, v, bias, km, dout = inputs(n, h, sq, sk, d, 0)
    scale = float(d) ** -0.5
    truth = reference_grads(q, k, v, bias, km, dout, scale)

    def loss(a, b, c, e):
      out = reference(a, b, c, e, km, scale)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    xla = {}
    for tag, dt in (('half', jnp.float16), ('float32', jnp.float32)):
      args = [x.astype(dt) for x in (q, k, v, bias)]
      xla[tag] = (reference(*args, km, scale),) + jax.grad(loss, argnums=(0, 1, 2, 3))(*args)

    # 32x32 is the one forward tiling every head dim has
    out, lse = fwd(q, k, v, bias, km, scale, 32, 32)
    ours = (out,) + bwd(q, k, v, bias, km, out, dout, lse, scale)
    names = ('out', 'dq', 'dk', 'dv', 'dbias')
    want = (truth[0],) + truth[2:]
    print(f'    n={n} h={h} sq={sq} sk={sk} D={d}')
    for i, name in enumerate(names):
      ro, rh, rf = (rel(x[i], want[i]) for x in (ours, xla['half'], xla['float32']))
      flag = '' if ro <= max(rh * 2, 1e-3) else '   WORSE THAN HALF XLA'
      bad += bool(flag)
      print(f'      {name:6s} ours {ro:.2e}   xla half {rh:.2e}   xla float32 {rf:.2e}{flag}')
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def dispatch_case():
  """The backward maps its grid one way below a batch of 64 and the other way at
  or above it, so both sides of that threshold need exercising."""
  print('either side of the grid-order threshold')
  bad = 0
  for n in (63, 64, 96):
    worst = try_run(n=n, h=2, sq=64, sk=64, d=32, quiet=True)
    flag = '' if worst is not None and worst < 0.05 else '  FAIL'
    bad += bool(flag)
    print(f'  b={n:3d}   worst {worst:.2e}{flag}')
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def want_flags_case():
  """want_lse and want_dbias only drop results, they never change the rest."""
  print('want_lse / want_dbias off')
  q, k, v, bias, km, dout = inputs(2, 2, 96, 96, 32, 0)
  scale = 32.0 ** -0.5
  out, lse = fwd(q, k, v, bias, km, scale)
  out_off, _ = fwd(q, k, v, bias, km, scale, want_lse=False)
  grads = bwd(q, k, v, bias, km, out, dout, lse, scale)
  off = bwd(q, k, v, bias, km, out, dout, lse, scale, want_dbias=False)
  bad = 0
  if not np.array_equal(np.asarray(out), np.asarray(out_off)):
    print('  out changed when want_lse was off')
    bad += 1
  for name, a, b in zip(('dk', 'dv'), grads[1:3], off[1:3]):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
      print(f'  {name} changed when want_dbias was off')
      bad += 1
  # dq arrives by atomic add, so it repeats only to a bound, never bit for bit
  r = rel(grads[0], off[0])
  print(f'  dq with dbias off, rel {r:.2e}')
  bad += r >= 1e-3
  if np.abs(np.asarray(off[3])).max() != 0.0:
    print('  dbias was written although want_dbias was off')
    bad += 1
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


if __name__ == '__main__':
  if len(sys.argv) > 1:
    FAMILY = sys.argv[1]
    assert FAMILY in ('mma', 'wmma')
  print('family:', FAMILY)
  register()
  bad = 0
  for kwargs in (dict(sq=96, sk=96, d=32),
                 dict(sq=96, sk=96, d=32, bq=64, bk=64),
                 dict(sq=64, sk=64, d=64, bk=64),
                 dict(sq=70, sk=83, d=16),        # ragged, both axes
                 dict(sq=128, sk=32, d=8, n=1, h=1)):
    print(kwargs)
    bad += check(str(kwargs), try_run(**kwargs))
  bad += masked_rows_case()
  bad += coverage_case()
  bad += amplitude_case()
  bad += tiny_case()
  bad += repeat_case()
  bad += vjp_case()
  bad += xla_case()
  bad += dispatch_case()
  bad += want_flags_case()
  print('RESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  sys.exit(0 if not bad else 1)
