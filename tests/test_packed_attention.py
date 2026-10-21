"""VoltaMmaPacked / VoltaWmmaPacked (and their Trim forms) against a float32 reference.

  KERNEL_DIR=<dir with libvolta_*.so> python tests/test_packed_attention.py [mma|wmma]

Rows and columns, heads 8 to 64, lengths that are and are not a multiple of 8 (on sm_70 they take the m8n8k4 kernel or
the wmma one), masked keys. Each shape runs in a fresh process with one allocation per array
(XLA_PYTHON_CLIENT_ALLOCATOR=platform), where a read past an array faults. Trim: live rows and keys as the reference,
the padded queries of a live row as without trim or zero, a row with no live key zero.
"""
import ctypes
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

KERNEL_DIR = os.environ.get('KERNEL_DIR', '.')
NEG = -1.0e9


def register(family):
  lib = ctypes.cdll.LoadLibrary(os.path.join(KERNEL_DIR, f'libvolta_{family}.so'))
  name = 'Mma' if family == 'mma' else 'Wmma'
  for sym in (f'Volta{name}Packed', f'Volta{name}PackedTrim'):
    jax.ffi.register_ffi_target(sym, jax.ffi.pycapsule(getattr(lib, sym)), platform='CUDA')
  return f'Volta{name}Packed'


def call(sym, p, bias, kmask, gb, column, lims=None):
  X, Y, _, H, D = p.shape
  args = (p, bias, kmask.astype(jnp.uint8), gb) + (() if lims is None else (lims,))
  return jax.ffi.ffi_call(sym + ('' if lims is None else 'Trim'), jax.ShapeDtypeStruct((X, Y, H, D), jnp.float16),
                          vmap_method='sequential')(*args, scale=np.float32(D ** -.5), block_q=np.int64(64),
                                                    block_k=np.int64(64 if sym == 'VoltaMmaPacked' and D == 64 else 32),
                                                    column=column)  # as volta.py: mma has no (64, 64, 32)


def reference(p, bias, kmask, gb, column):
  """[batch, sequence, H, D] in float32; batch along X for rows, Y for columns."""
  p = np.asarray(p, np.float32)
  if column:
    p = np.transpose(p, (1, 0, 2, 3, 4))
  q, k, v, g = (jnp.asarray(p[:, :, i]) for i in range(4))
  hp = jax.lax.Precision.HIGHEST
  logits = jnp.einsum('bqhd,bkhd->bhqk', q, k, precision=hp) * q.shape[-1] ** -.5 + jnp.asarray(bias, jnp.float32)[None]
  logits = jnp.where(jnp.asarray(kmask)[:, None, None, :], logits, NEG)
  o = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, -1), v, precision=hp)
  return np.asarray(o * jax.nn.sigmoid(g + jnp.asarray(gb, jnp.float32)[None, None]))


def trim_limits(kmask):
  """Each batch row's last live key + 1, then the largest: the lims the Trim form takes."""
  s = kmask.shape[1]
  last = np.where(kmask.any(axis=1), s - np.argmax(kmask[:, ::-1], axis=1), 0)
  return jnp.asarray(np.concatenate([last, [last.max()]]), jnp.int32)


def one(family, X, Y, H, D, column, frac, live):
  sym = register(family)
  rng = np.random.default_rng(0)
  p = jnp.asarray(rng.normal(size=(X, Y, 4, H, D)), jnp.float16)
  N, S = (Y, X) if column else (X, Y)
  bias = jnp.asarray(rng.normal(size=(H, S, S)), jnp.float16)
  gb = jnp.asarray(rng.normal(size=(H, D)) * .1, jnp.float16)
  km = rng.random((N, S)) >= frac
  if live < S:  # a padded tail and a dead last row
    km[:, live:] = False
    km[-1] = False
  r = reference(p, bias, km, gb, column)
  to_bs = (lambda o: np.transpose(np.asarray(o, np.float32), (1, 0, 2, 3))) if column else (
      lambda o: np.asarray(o, np.float32))
  o = to_bs(call(sym, p, bias, jnp.asarray(km), gb, column))
  rows = km.any(axis=1)
  err = np.abs(o - r)[rows].max() if rows.any() else 0.0
  if live >= S:
    return err < 0.03, f'max err {err:.2e}'
  t = to_bs(call(sym, p, bias, jnp.asarray(km), gb, column, trim_limits(km)))
  e_live = np.abs(t[rows][:, :live] - r[rows][:, :live]).max()
  pad = t[rows][:, live:]
  e_pad = np.minimum(np.abs(pad - o[rows][:, live:]), np.abs(pad)).max()
  e_dead = np.abs(t[~rows]).max()
  return (err < 0.03 and e_live < 0.03 and e_pad == 0 and e_dead == 0,
          f'max err {err:.2e}, trim live {e_live:.2e} padded {e_pad:.2e} dead {e_dead:.2e}')


# X, Y, H, D, column, masked fraction, live length (below the sequence: trim)
CASES = ((384, 384, 4, 32, False, 0.01, 384), (96, 384, 8, 32, True, 0.01, 96), (64, 256, 8, 8, False, 0.05, 256),
         (256, 64, 8, 8, True, 0.05, 64), (16, 128, 4, 16, False, 0.3, 128), (12, 200, 4, 64, False, 0.01, 200),
         (1, 261, 8, 32, True, 0.0, 1), (261, 1, 8, 32, False, 0.0, 1), (3, 37, 4, 32, True, 0.2, 3),
         (4, 264, 4, 32, False, 0.05, 264), (5, 129, 4, 32, True, 0.5, 5), (6, 256, 4, 32, False, 0.05, 200),
         (256, 6, 4, 32, True, 0.05, 200), (4, 128, 8, 8, False, 0.05, 72), (4, 261, 4, 32, False, 0.05, 200))

if __name__ == '__main__':
  if len(sys.argv) > 2 and sys.argv[2] == 'one':
    ok, msg = one(sys.argv[1], *[int(a) for a in sys.argv[3:7]], sys.argv[7] == '1', float(sys.argv[8]),
                  int(sys.argv[9]))
    print(msg)
    sys.exit(0 if ok else 1)
  family = sys.argv[1] if len(sys.argv) > 1 else 'mma'
  fails = 0
  for c in CASES:
    args = [str(int(x)) if isinstance(x, bool) else str(x) for x in c]
    run = subprocess.run([sys.executable, __file__, family, 'one', *args], capture_output=True, text=True,
                         env={**os.environ, 'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'})
    out = (run.stdout + run.stderr).strip().splitlines()
    ok = run.returncode == 0
    fails += not ok
    print(f"{'ok  ' if ok else 'BAD '}{family} X Y H D column masked live = {c}: {(out[-1] if out else '')[:140]}",
          flush=True)
  print('FAIL', fails) if fails else print('PASS')
  sys.exit(1 if fails else 0)
