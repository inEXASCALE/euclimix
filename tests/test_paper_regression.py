"""CUDA regression tests against the reference implementation."""

import os
from pathlib import Path

import pytest
import torch

import euclimix

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def original():
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    if os.environ.get("EUCLIMIX_COMPARE_PAPER") != "1":
        pytest.skip("set EUCLIMIX_COMPARE_PAPER=1 to compile reference sources")
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / "reference"
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield load(
            name="euclimix_reference_comparison",
            sources=[str(root / name) for name in ["euclidean_cuda_mix.cpp", "euclidean_kernels_mix.cu", "euclidean_kernels.cu"]],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_ldflags=["-lcublas"],
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("precision", [
    "fp16",
    pytest.param("bf16", marks=pytest.mark.skipif(
        "torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8",
        reason="BF16 requires SM80+",
    )),
    "fp32",
])
@pytest.mark.parametrize("correction", ["same", "fp32", "fp64"])
def test_reference_precision_and_selection(original, precision, correction):
    torch.manual_seed(12)
    x, y = torch.randn(17, 128, device="cuda"), torch.randn(11, 128, device="cuda")
    y[0] = x[0] + 1e-6
    # The reference implementation requires the default CUDA stream.
    actual = euclimix.pairwise_squared_distances(x, y, precision=precision, correction=correction, return_info=True)
    high = precision if correction == "same" else correction
    fn = getattr(original, f"pairwise_euclidean_{precision}_{high}_with_stats")
    expected, count = fn(x, y, 5.)
    torch.testing.assert_close(actual.distances, expected, rtol=2e-5, atol=1e-6)
    assert actual.fallback_count == count

