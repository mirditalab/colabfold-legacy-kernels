# colabfold-legacy-kernels

Prebuilt CUDA kernels to make ColabFold fast on Volta and Turning NVIDIA GPUs.
This package has three kernels that replace the Ampere+ Pallas/Triton kernels:
Attention, layer norm and the gated dual projection.

```python
import colabfold_legacy_kernels as clk
clk.available(cc=70)                  # True if the wheel has kernels for sm_70
clk.library_path("attention", cc=70)  # path of the shared library
clk.symbol("attention", cc=70)        # XLA FFI target name to register
```

To build the kernels, run the build script. It needs nvcc and the XLA FFI
headers, and fetches CUTLASS itself.

```bash
NVCC=/usr/local/cuda-12.9/bin/nvcc scripts/build_all.sh   # sm_70 and sm_75
ARCH=75 scripts/build_kernels.sh                          # one architecture
```
