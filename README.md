# euclimix

Mixed-precision Euclidean distances on NVIDIA GPUs, with Python/pybind11 and
C++/CUDA interfaces. Based on [Mixed-Precision Lloyd's Algorithm for k-means
Clustering](https://arxiv.org/abs/2407.12208).

## Installation

Requirements: Linux, Python 3.11+, CUDA-enabled PyTorch 2.11+, CUDA toolkit
12.8+ with `nvcc`, and an Ampere or newer GPU (SM80+). A compatible C++17
compiler is required; newer LibTorch versions may require C++20.
The PyTorch CUDA wheel provides runtime libraries, not the CUDA compiler.
cuDNN is not required.

Example with CUDA 12.8:

```bash
conda create -n euclimix python=3.12 -y
conda activate euclimix
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python -c 'import euclimix; euclimix.build_cuda()'
```

Set `CUDA_HOME` if the toolkit is not detected. Extensions compile on first
use and are cached by PyTorch. Set `TORCH_CUDA_ARCH_LIST` for the target GPU
(for example, `8.0` for A100), `MAX_JOBS` to limit build parallelism, and
`EUCLIMIX_BUILD_VERBOSE=1` for compiler output. Build and warm up the functions
before timing them.

## Python interface

```python
import torch
from euclimix import (
    pairwise_squared_distances, pairwise_distances,
    controlled_squared_distances, direct_squared_distances,
)

X = torch.randn(1000, 128, device='cuda', dtype=torch.float32)
Y = torch.randn(500, 128, device='cuda', dtype=torch.float32)

D = pairwise_squared_distances(X, Y, precision='fp16', correction='fp64')
distances = pairwise_distances(X, Y)
info = controlled_squared_distances(
    X, Y, precision='bf16', storage='fp32', test='extended', return_info=True,
)
reference = direct_squared_distances(X, Y, precision='fp64')
print(info.fallback_fraction)
```

| Function | Result | Implementation |
|---|---|---|
| `pairwise_squared_distances` | FP32 squared distances | Expanded formula, Original test, index-list recomputation |
| `controlled_squared_distances` | FP32 squared distances | Configurable storage and Original/Extended test, mask-based recomputation |
| `direct_squared_distances` | FP32 or FP64 squared distances | Direct subtraction and FMA accumulation |
| `pairwise_distances` | FP32 Euclidean distances | Square root of `pairwise_squared_distances` |

Inputs must be finite, dense FP32 CUDA matrices on the same device. Omit `Y`
to compare `X` with itself. Noncontiguous inputs and negative views are
materialized. Empty row sets are supported; zero features and more than
`2**31-1` pairs are rejected. Autograd and CUDA graph capture are unsupported.

FP16/BF16 cross products use FP32 accumulation. FP32/FP64 recomputation reads
the original FP32 coordinates; mixed outputs are stored in FP32. The
`correction='same'` option uses quantized coordinates and low-format arithmetic,
which can lose accuracy or overflow. Finite inputs can also produce squared
distances outside the FP32 output range.

The Extended error bound assumes a bound on the inner-product arithmetic;
its cuBLAS application is conditional on that assumption. A relative-error
bound for distances does not ensure identical nearest-center assignments.
See [Numerical implementation](docs/implementation.md) for the precision paths
and threshold definitions.

Input validation and reading fallback statistics synchronize with the GPU.
Native kernels use the current CUDA stream. Callers using multiple streams
must establish producer/consumer dependencies with events or `wait_stream`.
The FP32 index-list path follows PyTorch's TF32/matrix-multiplication policy;
autocast does not override explicit native precision choices.

## C++/CUDA interface

The native API uses LibTorch CUDA tensors and the types in
`euclimix/distance.h`. See [examples/native.cpp](examples/native.cpp).

```bash
cmake -S src/euclimix/csrc -B .native-build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DEUCLIMIX_BUILD_EXAMPLE=ON
cmake --build .native-build -j2
.native-build/euclimix_native_example
.native-build/euclimix_state_example
```

Native inputs must be contiguous with negative views resolved. The CMake
library can be used without Python or pybind11.

## Testing

```bash
python -m pip install -e '.[test,build]'
python -m pytest -m 'not cuda' --cov=euclimix --cov-branch --cov-report=term-missing
EUCLIMIX_REQUIRE_CUDA=1 python -m pytest -m cuda
EUCLIMIX_REQUIRE_CUDA=1 EUCLIMIX_COMPARE_PAPER=1 python -m pytest -m cuda
```

CPU CI requires 100% Python statement and branch coverage. CUDA integration
tests require a GPU; the GPU workflow uses a self-hosted Linux runner labeled
`self-hosted, linux, x64, gpu`. Python coverage does not measure CUDA code.
See [Testing](docs/validation.md) for device-memory checks and runner setup.

Reference CUDA sources in `tests/reference` are used for regression tests.
Their source hashes are recorded in [paper_sources.json](docs/paper_sources.json).
