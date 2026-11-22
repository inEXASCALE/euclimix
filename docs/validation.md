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
