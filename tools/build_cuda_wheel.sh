#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p -- "${1:-$repo/dist}"
wheel_dir=$(cd -- "${1:-$repo/dist}" && pwd)
raw_dir=$(mktemp -d)
trap 'rm -rf -- "$raw_dir"' EXIT

EUCLIMIX_BUILD_CUDA_WHEEL=1 python -m build --wheel --no-isolation --outdir "$raw_dir" "$repo"
python -m auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --strip \
    --wheel-dir "$raw_dir/repaired" \
    --exclude 'libtorch*.so' --exclude 'libc10*.so' \
    --exclude 'libcublas.so.13' --exclude 'libcudart.so.13' "$raw_dir"/*.whl
python "$repo/tools/check_cuda_wheel.py" "$raw_dir/repaired"/*.whl
cp -- "$raw_dir/repaired"/*.whl "$wheel_dir/"
