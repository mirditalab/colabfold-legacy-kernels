"""VoltaMma / VoltaMmaBwd against JAX's own autodiff, and the wmma pair.

  KERNEL_DIR=<dir with libvolta_*.so> python tests/test_attn_bwd.py [mma|wmma]

Neither family needs the card it targets: the mma kernels run on any
Turing-or-newer card and the wmma ones on anything from Volta up, so both can
be developed on an Ampere box and only confirmed on the T4 or V100 they are
meant for.
"""
import ctypes
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
  return jax.ffi.ffi_call(
      'VoltaMma' if FAMILY == 'mma' else 'VoltaWmma',
      (jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
       jax.ShapeDtypeStruct((n, h, sq), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, scale=np.float32(scale),
          block_q=np.int64(bq), block_k=np.int64(bk), want_lse=want_lse)


def bwd(q, k, v, bias, kmask, dout, lse, delta, scale, bq=64, bk=32,
        want_dbias=True):
  n, h, sq, d = q.shape
  sk = k.shape[2]
  return jax.ffi.ffi_call(
      'VoltaMmaBwd' if FAMILY == 'mma' else 'VoltaWmmaBwd',
      (jax.ShapeDtypeStruct(q.shape, jnp.float16),
       jax.ShapeDtypeStruct(k.shape, jnp.float16),
       jax.ShapeDtypeStruct(v.shape, jnp.float16),
       jax.ShapeDtypeStruct((h, sq, sk), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, dout, lse, delta, scale=np.float32(scale),
          block_q=np.int64(bq), block_k=np.int64(bk), want_dbias=want_dbias)


def reference(q, k, v, bias, kmask, scale):
  """fp32 attention with the kernel's own masking convention."""
  logits = scale * jnp.einsum('nhqd,nhkd->nhqk', q, k) + bias[None]
  logits = jnp.where(kmask[:, None, None, :] != 0, logits, NEG)
  p = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('nhqk,nhkd->nhqd', p, v)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float('inf')          # a NaN must never max() away behind a number
  denom = max(np.abs(b).max(), 1e-6)
  return float(np.abs(a - b).max() / denom)


def inputs(n, h, sq, sk, d, seed, masked_rows=0.15):
  ks = jax.random.split(jax.random.PRNGKey(seed), 6)
  f16 = lambda x: x.astype(jnp.float16)
  q = f16(jax.random.normal(ks[0], (n, h, sq, d)) * 0.5)
  k = f16(jax.random.normal(ks[1], (n, h, sk, d)) * 0.5)
  v = f16(jax.random.normal(ks[2], (n, h, sk, d)) * 0.5)
  bias = f16(jax.random.normal(ks[3], (h, sq, sk)) * 0.5)
  kmask = (jax.random.uniform(ks[4], (n, sk)) > masked_rows).astype(jnp.uint8)
  dout = f16(jax.random.normal(ks[5], (n, h, sq, d)) * 0.5)
  return q, k, v, bias, kmask, dout


def run(n=3, h=4, sq=96, sk=96, d=32, seed=0, bq=64, bk=32, kmask=None,
        quiet=False):
  q, k, v, bias, km, dout = inputs(n, h, sq, sk, d, seed)
  if kmask is not None:
    km = kmask
  scale = float(d) ** -0.5

  qf, kf, vf, bf = [x.astype(jnp.float32) for x in (q, k, v, bias)]
  ref_out, vjp = jax.vjp(
      lambda a, b, c, e: reference(a, b, c, e, km, scale), qf, kf, vf, bf)
  ref_dq, ref_dk, ref_dv, ref_dbias = vjp(dout.astype(jnp.float32))

  out, lse = fwd(q, k, v, bias, km, scale, bq, bk)
  delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), -1)
  dq, dk, dv, dbias = bwd(q, k, v, bias, km, dout, lse, delta, scale, bq, bk)

  # lse comes back in the log2 domain, which is what the kernel's exp2 wants.
  ref_logits = scale * jnp.einsum('nhqd,nhkd->nhqk', qf, kf) + bf[None]
  ref_logits = jnp.where(km[:, None, None, :] != 0, ref_logits, NEG)
  ref_lse2 = jax.scipy.special.logsumexp(ref_logits, -1) * 1.4426950408889634

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


def check(label, worst, tol=0.05):
  ok = worst < tol
  print('  ->', 'OK' if ok else 'FAIL', f'({label} worst {worst:.2e})')
  return not ok


def masked_rows_case():
  """A batch element with every key masked: the forward softmax is uniform."""
  print('one batch element with every key masked')
  km = np.ones((3, 96), np.uint8)
  km[1, :] = 0
  worst = run(kmask=jnp.asarray(km))
  return check('all-masked', worst)


def coverage_case():
  """Whatever block shape the forward takes, the backward must differentiate."""
  print('every forward (D, block_q, block_k)')
  bad = 0
  for d, bq, bk in FWD_CONFIGS[FAMILY]:
    try:
      worst = run(n=2, h=2, sq=130, sk=130, d=d, bq=bq, bk=bk, quiet=True)
    except Exception as exc:                 # noqa: BLE001 - report, keep going
      print(f'  D={d:3d} bq={bq:3d} bk={bk:3d}  {type(exc).__name__}: '
            f'{str(exc).splitlines()[0][:60]}')
      bad += 1
      continue
    flag = '' if worst < 0.05 else '  FAIL'
    print(f'  D={d:3d} bq={bq:3d} bk={bk:3d}  worst {worst:.2e}{flag}')
    bad += worst >= 0.05
  print('  ->', 'OK' if not bad else f'FAIL ({bad})')
  return bad


def want_flags_case():
  """want_lse and want_dbias only drop results, they never change the rest."""
  print('want_lse / want_dbias off')
  q, k, v, bias, km, dout = inputs(2, 2, 96, 96, 32, 0)
  scale = 32.0 ** -0.5
  out, lse = fwd(q, k, v, bias, km, scale)
  out_off, _ = fwd(q, k, v, bias, km, scale, want_lse=False)
  delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), -1)
  grads = bwd(q, k, v, bias, km, dout, lse, delta, scale)
  off = bwd(q, k, v, bias, km, dout, lse, delta, scale, want_dbias=False)
  bad = 0
  if not np.array_equal(np.asarray(out), np.asarray(out_off)):
    print('  out changed when want_lse was off')
    bad += 1
  for name, a, b in zip(('dq', 'dk', 'dv'), grads, off):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
      print(f'  {name} changed when want_dbias was off')
      bad += 1
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
    bad += check(str(kwargs), run(**kwargs))
  bad += masked_rows_case()
  bad += coverage_case()
  bad += want_flags_case()
  print('RESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  sys.exit(0 if not bad else 1)
