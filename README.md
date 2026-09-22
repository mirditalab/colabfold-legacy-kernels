# colabfold-legacy-kernels

Prebuilt CUDA kernels to make ColabFold fast on Volta and Turing NVIDIA GPUs.
This package has three kernels that replace the Ampere+ Pallas/Triton kernels:
Attention, layer norm and the gated dual projection.

```python
import colabfold_legacy_kernels as clk
clk.available(cc=70)                  # True if the wheel has kernels for sm_70
clk.library_path("attention", cc=70)  # path of the shared library
clk.symbol("attention", cc=70)        # XLA FFI target name to register
```

The attention kernel also has a backward pass on both architectures, so a model
can be trained or designed against it. It returns dQ, dK, dV and dBias. MSA
attention takes q, k and v from the MSA and the pair representation only as
that bias, so dBias is the one path a gradient has back to the pair trunk.

```python
clk.symbol("attention_bwd", cc=75)  # VoltaMmaBwd   dq, dk, dv, dbias
clk.symbol("attention_bwd", cc=70)  # VoltaWmmaBwd  the same, with wmma
```

The backward rebuilds the softmax from `lse`, the second result of the forward.
Pass `want_lse=False` and a one element buffer for it when only inferring, and
`want_dbias=False` when the bias takes no gradient.

To build the kernels, run the build script. It needs nvcc and the XLA FFI
headers, and fetches CUTLASS itself.

```bash
NVCC=/usr/local/cuda-12.9/bin/nvcc scripts/build_all.sh   # sm_70 and sm_75
ARCH=75 scripts/build_kernels.sh                          # one architecture
```
