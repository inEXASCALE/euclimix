"""CUDA integration tests."""

import itertools

import pytest
import torch

import euclimix

pytestmark = pytest.mark.cuda
requires_bf16 = pytest.mark.skipif(
    "torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8",
    reason="BF16 requires SM80+",
)
BF16 = pytest.param("bf16", marks=requires_bf16)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        euclimix.build_cuda()
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def data(n=7, m=5, r=19):
    g = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(n, r, generator=g, device="cuda")
    y = torch.randn(m, r, generator=g, device="cuda")
    if n and m:
        y[0] = x[0] + 1e-6
    return x, y


def reference(x, y):
    return ((x.double()[:, None] - y.double()[None]) ** 2).sum(-1)


@pytest.mark.parametrize("precision", ["fp16", BF16, "fp32"])
@pytest.mark.parametrize("correction", ["same", "fp32", "fp64"])
def test_paper_all_modes(precision, correction):
    x, y = data()
    result = euclimix.pairwise_squared_distances(x, y, precision=precision, correction=correction, return_info=True)
    expected = reference(x, y)
    assert result.distances.dtype == torch.float32
    assert result.distances.shape == (7, 5)
    assert result.fallback_mask.dtype == torch.bool
    assert result.fallback_mask[0, 0]
    assert torch.isfinite(result.distances).all()
    assert (result.distances >= 0).all()
    euclidean = euclimix.pairwise_distances(x, y, precision=precision, correction=correction)
    assert torch.equal(euclidean, result.distances.sqrt())
    # Same-format correction deliberately uses quantized inputs.
    if correction != "same":
        tolerance = 3e-2 if precision == "bf16" else 3e-3
        torch.testing.assert_close(result.distances.double(), expected, rtol=tolerance, atol=2e-5)
        torch.testing.assert_close(result.distances[result.fallback_mask], expected.float()[result.fallback_mask], rtol=3e-6, atol=1e-12)


@pytest.mark.parametrize("precision", ["fp16", BF16])
@pytest.mark.parametrize("storage,test,backend,correction", list(itertools.product(
    ["low", "fp32"], ["original", "extended"], ["cublas", "sequential"], ["fp32", "fp64"],
)))
def test_controlled_all_modes(precision, storage, test, backend, correction):
    x, y = data()
    result = euclimix.controlled_squared_distances(x, y, precision=precision, storage=storage, test=test, backend=backend, correction=correction, return_info=True)
    expected = reference(x, y)
    assert result.fallback_mask[0, 0]
    assert torch.isfinite(result.absolute_bound).all()
    assert torch.isfinite(result.distances).all()
    accepted = ~result.fallback_mask
    error = (result.distances.double() - expected).abs()
    assert torch.all(error[accepted] <= result.absolute_bound[accepted] + expected[accepted].abs() * 1e-12)
    if test == "extended":
        assert torch.all(error[accepted] <= expected[accepted] / 4 + 1e-12)
    torch.testing.assert_close(result.distances[result.fallback_mask], expected.float()[result.fallback_mask], rtol=3e-6, atol=1e-12)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_direct(precision):
    x, y = data()
    out = euclimix.direct_squared_distances(x, y, precision=precision)
    torch.testing.assert_close(out.double(), reference(x, y), rtol=3e-6 if precision == "fp32" else 1e-14, atol=1e-14)
    assert out.dtype == (torch.float64 if precision == "fp64" else torch.float32)


@pytest.mark.parametrize("function", [euclimix.pairwise_squared_distances, euclimix.controlled_squared_distances, euclimix.direct_squared_distances, euclimix.pairwise_distances])
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 0)])
def test_empty(function, shape):
    x, y = data(*shape)
    assert function(x, y).shape == shape


@pytest.mark.parametrize("function", [euclimix.pairwise_squared_distances, euclimix.controlled_squared_distances])
def test_noncontiguous_self_and_diagonal(function):
    x, _ = data(n=9, r=18)
    x = x[:, ::2]
    assert not x.is_contiguous()
    out = function(x)
    assert torch.count_nonzero(out.diag()) == 0
    # Library reductions need not produce bitwise symmetric low-format outputs.
    torch.testing.assert_close(out, out.t(), rtol=4e-3, atol=1e-6)


@pytest.mark.parametrize("function", [euclimix.pairwise_squared_distances, euclimix.controlled_squared_distances])
def test_conversion_overflow_recomputed_from_original(function):
    x = torch.tensor([[100000., -100000.]], device="cuda")
    y = x + 1
    result = function(x, y, precision="fp16", correction="fp64", return_info=True)
    assert result.fallback_count == 1
    assert result.distances.item() == 2


def test_euclidean_and_zero_vectors():
    x = torch.tensor([[0., 0.]], device="cuda")
    y = torch.tensor([[3., 4.]], device="cuda")
    assert euclimix.pairwise_distances(x, y).item() == 5
    assert euclimix.pairwise_squared_distances(x).item() == 0


def test_euclidean_sqrt_at_numeric_extremes():
    x = torch.tensor([[2.0**-63], [1e-22], [1e20]], device="cuda")
    y = torch.zeros((1, 1), device="cuda")
    squared = euclimix.pairwise_squared_distances(x, y, correction="fp64")
    assert squared[0, 0].item() == torch.finfo(torch.float32).tiny
    assert torch.isinf(squared[2, 0])
    assert torch.equal(euclimix.pairwise_distances(x, y, correction="fp64"), squared.sqrt())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_paper_rejects_nonfinite_coordinates(bad):
    empty = torch.empty((0, 1), device="cuda")
    invalid = torch.tensor([[bad]], device="cuda")
    paper, _ = euclimix.build_cuda()
    for x, y in ((empty, invalid), (invalid, empty)):
        with pytest.raises(ValueError, match="finite coordinates"):
            euclimix.pairwise_squared_distances(x, y)
        with pytest.raises(ValueError, match="finite coordinates"):
            euclimix.pairwise_distances(x, y)
        with pytest.raises(ValueError, match="finite coordinates"):
            paper.distance(x, y, 0, 32, 5.)
        with pytest.raises(ValueError, match="finite coordinates"):
            paper.euclidean(x, y, 0, 32, 5.)


def test_controlled_gradual_underflow():
    x = torch.tensor([[1e-22]], device="cuda")
    y = torch.zeros_like(x)
    result = euclimix.controlled_squared_distances(x, y, return_info=True)
    assert result.fallback_count == 1
    assert result.distances.item() > 0
    assert result.distances.item() == reference(x, y).float().item()


@requires_bf16
def test_bf16_original_full_recomputation_at_paper_dimension():
    x, y = data(n=5, m=3, r=128)
    for storage in ("low", "fp32"):
        result = euclimix.controlled_squared_distances(x, y, precision="bf16", storage=storage, test="original", return_info=True)
        assert result.fallback_count == 15
        torch.testing.assert_close(result.distances, reference(x, y).float())


@pytest.mark.parametrize("precision", ["fp16", BF16])
@pytest.mark.parametrize("test", ["original", "extended"])
def test_exact_near_and_far_selection(precision, test):
    # Exactly representable inputs and dot products give independent expected
    # decisions. An implementation that always recomputes must fail this test.
    x = torch.ones(1, 128, device="cuda")
    y = torch.cat((x, -x), dim=0)
    expected_mask = [[True, test == "original" and precision == "bf16"]]
    expected_distances = torch.tensor([[0., 512.]], device="cuda")
    for storage, backend in itertools.product(["low", "fp32"], ["cublas", "sequential"]):
        result = euclimix.controlled_squared_distances(
            x, y, precision=precision, storage=storage, test=test,
            backend=backend, return_info=True,
        )
        assert result.fallback_mask.tolist() == expected_mask
        torch.testing.assert_close(result.distances, expected_distances)


@pytest.mark.parametrize("precision", ["fp16", BF16])
def test_fp32_storage_retains_small_norm_contribution(precision):
    # Both coordinates are exactly representable in both input formats.
    # 1 + 2**-12 is retained in FP32, but rounds to 1 in low norm storage.
    x = torch.tensor([[1., 2**-6]], device="cuda")
    y = torch.zeros_like(x)
    low = euclimix.controlled_squared_distances(x, y, precision=precision, storage="low", return_info=True)
    wide = euclimix.controlled_squared_distances(x, y, precision=precision, storage="fp32", return_info=True)
    assert low.fallback_count == wide.fallback_count == 0
    assert low.distances.item() == 1.
    assert wide.distances.item() == 1. + 2**-12


@pytest.mark.parametrize("precision", ["fp16", BF16])
def test_same_correction_uses_quantized_inputs(precision):
    x = torch.ones(1, 128, device="cuda")
    y = x + 2**-12  # Rounds back to 1 in either low input format.
    same = euclimix.pairwise_squared_distances(x, y, precision=precision, correction="same", return_info=True)
    high = euclimix.pairwise_squared_distances(x, y, precision=precision, correction="fp64", return_info=True)
    assert same.fallback_count == high.fallback_count == 1
    assert same.distances.item() == 0.
    assert high.distances.item() == 2**-17


@pytest.mark.parametrize("precision", ["fp16", BF16])
def test_same_correction_accumulates_in_low_format(precision):
    x = torch.full((1, 128), 16., device="cuda")
    y = x + 0.5
    y[0, 0] = 48.
    # Differences and products are exactly representable. After the first
    # square (1024), adding 0.25 rounds away in both FP16 and BF16, but not FP32.
    same = euclimix.pairwise_squared_distances(x, y, precision=precision, correction="same", return_info=True)
    high = euclimix.pairwise_squared_distances(x, y, precision=precision, correction="fp64", return_info=True)
    assert same.fallback_count == high.fallback_count == 1
    assert same.distances.item() == 1024.
    assert high.distances.item() == 1055.75


@pytest.mark.parametrize("function", [euclimix.pairwise_squared_distances, euclimix.controlled_squared_distances, euclimix.pairwise_distances])
def test_nondefault_stream(function):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, y = data(n=33, m=17, r=128)
        result = function(x, y)
        expected = reference(x, y)
        if function is euclimix.pairwise_distances:
            expected = expected.sqrt()
        done = stream.record_event()
    torch.cuda.current_stream().wait_event(done)
    torch.testing.assert_close(result.double(), expected, rtol=4e-3, atol=1e-5)


def test_autocast_does_not_change_explicit_precision():
    x, y = data()
    expected = euclimix.pairwise_squared_distances(x, y, precision="fp32")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = euclimix.pairwise_squared_distances(x, y, precision="fp32")
    torch.testing.assert_close(actual, expected)


def test_negative_views_are_materialized():
    x, y = data()
    expected = euclimix.pairwise_squared_distances(-x, y)
    actual = euclimix.pairwise_squared_distances(x._neg_view(), y)
    torch.testing.assert_close(actual, expected)


def test_native_validation_cannot_be_bypassed():
    paper, controlled = euclimix.build_cuda()
    x, y = data()
    with pytest.raises(RuntimeError, match="CUDA"):
        paper.distance(x.cpu(), y.cpu(), 0, 64, 5.)
    with pytest.raises(RuntimeError, match="FP32"):
        paper.distance(x.half(), y.half(), 0, 64, 5.)
    with pytest.raises(RuntimeError, match="correction"):
        paper.distance(x, y, 0, 17, 5.)
    with pytest.raises(RuntimeError, match="rho"):
        paper.distance(x, y, 0, 64, float("nan"))
    with pytest.raises(RuntimeError, match="invalid mode"):
        controlled.distance(x, y, 0, 0, 0, 0, 64, 32, 5., -1.)
    with pytest.raises(RuntimeError, match="precision"):
        controlled.direct(x, y, 17)


def test_turing_rejects_bf16():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("requires an SM75 GPU")
    x, y = data()
    paper, controlled = euclimix.build_cuda()
    with pytest.raises(RuntimeError, match="BF16.*SM80"):
        euclimix.pairwise_distances(x, y, precision="bf16")
    with pytest.raises(RuntimeError, match="BF16.*SM80"):
        euclimix.controlled_squared_distances(x, y, precision="bf16")
    with pytest.raises(RuntimeError, match="BF16.*SM80"):
        paper.distance(x, y, 1, 64, 5.)
    with pytest.raises(RuntimeError, match="BF16.*SM80"):
        controlled.distance(x, y, 1, 1, 0, 2, 64, 32, 5., -1.)


def test_device_guard():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    original = torch.cuda.current_device()
    other = (original + 1) % torch.cuda.device_count()
    x = torch.ones(2, 4, device=f"cuda:{other}")
    out = euclimix.pairwise_squared_distances(x)
    assert out.device == x.device
    assert torch.cuda.current_device() == original

