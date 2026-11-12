# euclimix

[![CI](https://img.shields.io/github/actions/workflow/status/inEXASCALE/euclid-cuda/cpu.yml?branch=main&label=CI&logo=githubactions&logoColor=white&style=flat-square)](https://github.com/inEXASCALE/euclid-cuda/actions/workflows/cpu.yml)
[![CUDA tests](https://img.shields.io/github/actions/workflow/status/inEXASCALE/euclid-cuda/gpu.yml?branch=main&label=CUDA%20tests&logo=nvidia&logoColor=white&style=flat-square)](https://github.com/inEXASCALE/euclid-cuda/actions/workflows/gpu.yml)
[![Python coverage gate: 100%](https://img.shields.io/badge/Python%20coverage%20gate-100%25-2C974B?logo=pytest&logoColor=white&style=flat-square)](docs/validation.md#python-tests)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](pyproject.toml)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white&style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv:2407.12208](https://img.shields.io/badge/arXiv-2407.12208-B31B1B?logo=arxiv&logoColor=white&style=flat-square)](#citation)

Mixed-precision Euclidean distances on NVIDIA GPUs, with Python/pybind11 and
C++/CUDA interfaces. Low-precision expanded distances are combined with
selective higher-precision recomputation.

## Installation

Requirements: Linux, Python 3.11+, CUDA-enabled PyTorch 2.11+, CUDA toolkit
12.8+ with `nvcc`, and an Ampere or newer GPU (SM80+). A compatible C++17
compiler is required; newer LibTorch versions may require C++20.
The PyTorch CUDA wheel provides runtime libraries, not the CUDA compiler.
cuDNN is not required.

Example with CUDA 12.8:

```bash
git clone https://github.com/inEXASCALE/euclid-cuda.git
cd euclid-cuda
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

### Squared and Euclidean distances

```python
import torch
from euclimix import pairwise_squared_distances, pairwise_distances

X = torch.tensor([[0., 0.], [3., 4.]], device='cuda', dtype=torch.float32)

squared = pairwise_squared_distances(X)
distances = pairwise_distances(X)

print(squared.cpu())    # tensor([[ 0., 25.], [25.,  0.]])
print(distances.cpu())  # tensor([[0., 5.], [5., 0.]])
```

Omitting `Y` computes all distances within `X`. Pass a second matrix to
compute an `(n_samples_X, n_samples_Y)` result.

### Precision and recomputation statistics

```python
import torch
from euclimix import pairwise_squared_distances

X = torch.randn(1024, 128, device='cuda', dtype=torch.float32)
Y = torch.randn(512, 128, device='cuda', dtype=torch.float32)

result = pairwise_squared_distances(
    X, Y, precision='fp16', correction='fp64', rho=5.0, return_info=True,
)

print(result.distances.shape)      # torch.Size([1024, 512])
print(result.distances.dtype)      # torch.float32
print(result.fallback_count)
print(f'Recomputed: {result.fallback_fraction:.2%}')
```

### Storage and threshold selection

```python
import torch
from euclimix import controlled_squared_distances, direct_squared_distances

X = torch.randn(256, 128, device='cuda', dtype=torch.float32)
Y = torch.randn(128, 128, device='cuda', dtype=torch.float32)

result = controlled_squared_distances(
    X, Y, precision='bf16', storage='fp32', test='extended', return_info=True,
)
reference = direct_squared_distances(X, Y, precision='fp64')
max_absolute_error = (result.distances.double() - reference).abs().max()

print(result.model)
print(result.fallback_fraction)
print(max_absolute_error.item())
```

Use `storage='low'` to store intermediates in the input format and
`test='original'` to select the empirical threshold. The FP64 direct
reference retains FP64 output.

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

```cpp
#include <euclimix/distance.h>

euclimix::DistanceResult distances(torch::Tensor X, torch::Tensor Y) {
    euclimix::PaperOptions options;
    options.precision = euclimix::InputPrecision::FP16;
    options.correction = euclimix::Correction::FP64;
    options.rho = 5.0;
    return euclimix::paper_squared_distances(X, Y, options);
}
```

Build the native library and examples:

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

## Citation

If you use euclimix in your research, please cite the accompanying
[paper](https://arxiv.org/abs/2407.12208):

```bibtex
@misc{carson2024computing,
  title         = {Computing $k$-means in mixed precision},
  author        = {Erin Carson and Xinye Chen and Xiaobo Liu},
  year          = {2024},
  eprint        = {2407.12208},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2407.12208},
  url           = {https://arxiv.org/abs/2407.12208}
}
```
