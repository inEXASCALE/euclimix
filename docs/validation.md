# Testing

Install test and build dependencies:

```bash
python -m pip install -e '.[test,build]'
```

## Python tests

```bash
python -m pytest -m 'not cuda' --cov=euclimix --cov-branch \
  --cov-report=term-missing --cov-report=xml
```

CI requires 100% Python statement and branch coverage. Unit tests cover
input validation, dispatch, extension loading and result statistics. Native
entry points are stubbed in these tests; CUDA arithmetic is tested separately.

## Precompiled CUDA wheel

Use a manylinux_2_28 x86-64 build environment with the target CPython
interpreter and a compatible C++20 compiler. Prepare a CUDA 13.2 build
prefix from NVIDIA's redistributable archives, or set `CUDA_HOME` to an
existing CUDA 13.2 Toolkit and add its `bin` directory to `PATH`.

```bash
python -m pip install --index-url https://pypi.org/simple \
  'torch==2.15.0' 'setuptools>=69' wheel build ninja auditwheel patchelf
export CUDA_HOME="$(mktemp -d)/cuda-13.2"
python tools/prepare_cuda_toolkit.py "$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
bash tools/build_cuda_wheel.sh
```

The build uses the installed PyPI PyTorch wheel and checks that both backends
contain SM75 and SM80 machine code. The wheel declares its PyTorch runtime
dependency and reuses LibTorch, cuBLAS, and the CUDA runtime from the installed
dependencies. Install the wheel in a fresh environment with a GPU and driver
and run the smoke test with a new JIT cache path.

```bash
TORCH_EXTENSIONS_DIR="$(mktemp -d)/jit" python tools/smoke_cuda_wheel.py
```

The manually dispatched **Linux CUDA wheels** workflow builds one artifact per
CPython version from 3.11 through 3.14. Its smoke jobs use a self-hosted Linux
x86-64 runner labeled `gpu` with Docker and NVIDIA Container Toolkit. Each
job installs the corresponding artifact in a clean Python slim container and
checks both CUDA backends on the GPU without `nvcc`. The workflow stores the
wheels as GitHub Actions artifacts.

## CUDA tests

Use a supported NVIDIA GPU, a compatible host compiler and the CUDA toolkit.

```bash
export TORCH_CUDA_ARCH_LIST=8.0  # Use 7.5 for RTX 2080 Ti.
export MAX_JOBS=2
python -c 'import euclimix; euclimix.build_cuda()'
EUCLIMIX_REQUIRE_CUDA=1 python -m pytest -m cuda
```

Tests cover precision/correction modes, storage, threshold selection, near
pairs, overflow, underflow, tensor views, autocast and nondefault streams.
FP32 matrix-multiplication tests disable TF32 temporarily and restore the
previous policy. BF16 tests require SM80+ and the device-guard test requires
two GPUs.

Compare with the reference CUDA implementation:

```bash
EUCLIMIX_REQUIRE_CUDA=1 EUCLIMIX_COMPARE_PAPER=1 python -m pytest -m cuda
```

Check device memory accesses:

```bash
EUCLIMIX_REQUIRE_CUDA=1 compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -m pytest -m cuda tests/test_cuda.py
```

The native CMake examples in the README check the C++ interface and cuBLAS
mode/stream preservation. Source checksums for the regression fixtures are
stored in `docs/paper_sources.json`.

## CI

The CPU workflow runs on pushes and pull requests, builds and installs a
wheel, and uploads test and coverage reports. The GPU workflow is manually
dispatched on a self-hosted Linux runner labeled `self-hosted, linux, x64, gpu`.
It requires CUDA 12.8, CMake 3.24+, compute-sanitizer and a compatible compiler.

`EUCLIMIX_REQUIRE_CUDA=1` makes a missing GPU an error. Python coverage does
not measure CUDA code.
