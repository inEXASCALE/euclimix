# euclimix

[![CI](https://img.shields.io/github/actions/workflow/status/inEXASCALE/euclimix/cpu.yml?branch=main&label=CI&logo=githubactions&logoColor=white&style=flat-square)](https://github.com/inEXASCALE/euclimix/actions/workflows/cpu.yml)
[![CUDA tests](https://img.shields.io/github/actions/workflow/status/inEXASCALE/euclimix/gpu.yml?branch=main&label=CUDA%20tests&logo=nvidia&logoColor=white&style=flat-square)](https://github.com/inEXASCALE/euclimix/actions/workflows/gpu.yml)
[![Python coverage gate: 100%](https://img.shields.io/badge/Python%20coverage%20gate-100%25-2C974B?logo=pytest&logoColor=white&style=flat-square)](docs/validation.md#python-tests)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](pyproject.toml)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white&style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv:2407.12208](https://img.shields.io/badge/arXiv-2407.12208-B31B1B?logo=arxiv&logoColor=white&style=flat-square)](#citation)

Mixed-precision Euclidean distances on NVIDIA GPUs, with Python/pybind11 and C++/CUDA interfaces. Low-precision expanded distances are combined with selective higher-precision recomputation.

## Installation

### Precompiled CUDA wheel

Install a compatible release wheel from a clean CPython environment. For a local
wheel, pass its filename to pip.

```bash
python -m pip install ./euclimix-0.1.1-cp314-cp314-manylinux_2_28_x86_64.whl
```

The binary builds target CPython 3.11–3.14 on Linux x86-64 with glibc 2.28+,
NVIDIA SM75 or SM80, and PyTorch 2.15.0 with CUDA 13.2. Choose the wheel tag
matching your Python version. Pip installs PyTorch and the CUDA runtime
dependencies. The machine needs a compatible NVIDIA driver. CUDA Toolkit,
Ninja, and a host compiler are needed only for building the wheel or compiling
from source. BF16 still requires SM80+. See [wheel build instructions](docs/validation.md#precompiled-cuda-wheel).

### Source installation

Requirements include Linux, Python 3.11+, CUDA-enabled PyTorch 2.11+, CUDA toolkit 12.8+ with `nvcc`, and an NVIDIA Turing or newer GPU (SM75+). BF16 requires Ampere or newer (SM80+). The PyTorch wheel must support the target GPU. Use a compatible C++17 compiler or C++20 if required by LibTorch. The PyTorch CUDA wheel provides runtime libraries. Install the CUDA compiler separately. cuDNN is optional.

Example with CUDA 12.8:

```bash
git clone https://github.com/inEXASCALE/euclimix.git
cd euclimix
conda create -n euclimix python=3.12 -y
conda activate euclimix
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python -c 'import euclimix; euclimix.build_cuda()'
```

Set `CUDA_HOME` if the toolkit is not detected. Extensions compile on first use and are cached by PyTorch. Set `TORCH_CUDA_ARCH_LIST` for the target GPU (`7.5` for RTX 2080 Ti or `8.0` for A100), `MAX_JOBS` to limit build parallelism, and `EUCLIMIX_BUILD_VERBOSE=1` for compiler output. Build and warm up the functions before timing them.

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

Omitting `Y` computes all distances within `X`. Pass a second matrix to compute an `(n_samples_X, n_samples_Y)` result.

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
    X, Y, precision='fp16', storage='fp32', test='extended', return_info=True,
)
reference = direct_squared_distances(X, Y, precision='fp64')
max_absolute_error = (result.distances.double() - reference).abs().max()

print(result.model)
print(result.fallback_fraction)
print(max_absolute_error.item())
```

Use `storage='low'` to store intermediates in the input format and `test='original'` to select the empirical threshold. The FP64 direct reference retains FP64 output.

| Function | Result | Implementation |
|---|---|---|
| `pairwise_squared_distances` | FP32 squared distances | Expanded formula with Original test and fused selective recomputation |
| `controlled_squared_distances` | FP32 squared distances | Configurable storage and Original/Extended test, mask-based recomputation |
| `direct_squared_distances` | FP32 or FP64 squared distances | Direct subtraction and FMA accumulation |
| `pairwise_distances` | FP32 Euclidean distances | Same algorithm with a fused FP32 square root |

Inputs must be finite, dense FP32 CUDA matrices on the same device. Omit `Y` to compare `X` with itself. Noncontiguous inputs and negative views are materialized. Empty row sets are supported; zero features and more than `2**31-1` pairs are rejected. Autograd and CUDA graph capture are unsupported.

FP16/BF16 cross products use FP32 accumulation. FP32/FP64 recomputation reads the original FP32 coordinates; mixed outputs are stored in FP32. The `correction='same'` option uses quantized coordinates and low-format arithmetic, which can lose accuracy or overflow. Finite inputs can also produce squared distances outside the FP32 output range.

The Extended error bound assumes a bound on the inner-product arithmetic; its cuBLAS application is conditional on that assumption. A relative-error bound for distances does not ensure identical nearest-center assignments. See [Numerical implementation](docs/implementation.md) for the precision paths and threshold definitions.

Input validation and reading fallback statistics synchronize with the GPU. The two pairwise APIs check both inputs together in one native CUDA pass with one host readback for nonempty inputs. Native kernels use the current CUDA stream. Callers using multiple streams must establish producer/consumer dependencies with events or `wait_stream`. The FP32 pairwise path follows PyTorch's matrix-multiplication policy. Explicit native precision choices take precedence over autocast.

## C++/CUDA interface

The native API uses LibTorch CUDA tensors and the types in `euclimix/distance.h`. See [examples/native.cpp](examples/native.cpp).

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

Build the native library and examples. For an RTX 2080 Ti, set `TORCH_CUDA_ARCH_LIST=7.5` and pass `-DCMAKE_CUDA_ARCHITECTURES=75`.

```bash
cmake -S src/euclimix/csrc -B .native-build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DEUCLIMIX_BUILD_EXAMPLE=ON
cmake --build .native-build -j2
.native-build/euclimix_native_example
.native-build/euclimix_state_example
```

Native inputs must be contiguous with negative views resolved. The CMake library can be used without Python or pybind11.

## Testing

```bash
python -m pip install -e '.[test,build]'
python -m pytest -m 'not cuda' --cov=euclimix --cov-branch --cov-report=term-missing
EUCLIMIX_REQUIRE_CUDA=1 python -m pytest -m cuda
EUCLIMIX_REQUIRE_CUDA=1 EUCLIMIX_COMPARE_PAPER=1 python -m pytest -m cuda
```

CPU CI requires 100% Python statement and branch coverage. CUDA integration tests require a GPU; the GPU workflow uses a self-hosted Linux runner labeled `self-hosted, linux, x64, gpu`. Python coverage does not measure CUDA code. See [Testing](docs/validation.md) for device-memory checks and runner setup.

Reference CUDA sources in `tests/reference` are used for regression tests. Their source hashes are recorded in [paper_sources.json](docs/paper_sources.json).

## Citation

If you use euclimix in your research, please cite the accompanying paper, accepted for publication in the *SIAM Journal on Scientific Computing*. [arXiv preprint](https://arxiv.org/abs/2407.12208v4)

```bibtex
@article{carsonMixedPrecisionLloyd,
  author  = {Erin Carson and Xinye Chen and Xiaobo Liu},
  title   = {Mixed-Precision {Lloyd}'s Algorithm for $k$-means Clustering},
  journal = {SIAM Journal on Scientific Computing},
  year    = {forthcoming},
  url     = {https://arxiv.org/abs/2407.12208v4}
}
```
