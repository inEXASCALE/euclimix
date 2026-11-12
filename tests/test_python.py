"""Unit tests for validation, dispatch and result statistics."""

from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import euclimix
from euclimix import _api, _extension, _validation


class DeviceTensor(torch.Tensor):
    """CPU-backed tensors with CUDA metadata for validation tests."""

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")


def fake_cuda(value):
    return torch.as_tensor(value, dtype=torch.float32).as_subclass(DeviceTensor)


@pytest.fixture
def capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))


def test_public_exports():
    assert euclimix.__version__ == "0.1.0"
    for name in euclimix.__all__:
        assert getattr(euclimix, name) is not None


def test_distribution_contains_native_sources():
    root = Path(euclimix.__file__).parent / "csrc"
    for name in ("paper.cu", "controlled.cu", "paper_bindings.cpp", "controlled_bindings.cpp", "common.h", "include/euclimix/distance.h", "CMakeLists.txt", "native_example.cpp", "state_example.cpp"):
        assert (root / name).is_file(), f"missing packaged native file: {name}"


def test_reference_source_integrity():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/paper_sources.json").read_text())
    assert len(manifest["original_files"]) == 18
    for record in manifest["original_files"]:
        assert hashlib.sha256((root / record["file"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("value", [None, [], 1, "wrong"])
def test_choice_invalid(value):
    with pytest.raises(ValueError, match="must be one of"):
        _validation.choice(value, {"fp16": 0}, "precision")


def test_choice_valid():
    assert _validation.choice("fp16", {"fp16": 0}, "precision") == 0


@pytest.mark.parametrize("rho", [True, False, "5", None, [], 1j])
def test_rho_type(rho):
    with pytest.raises(TypeError, match="real number"):
        _validation.validate_rho(rho)


@pytest.mark.parametrize("rho", [float("nan"), float("inf"), -float("inf"), -1, 0, 1, 1e6 + 1,
    pytest.param(10**10000, id="huge-positive"),
    pytest.param(-(10**10000), id="huge-negative")])
def test_rho_range(rho):
    with pytest.raises(ValueError, match="finite"):
        _validation.validate_rho(rho)


def test_rho_valid():
    assert _validation.validate_rho(5) == 5.0
    assert _validation.validate_rho(1e6) == 1e6


@pytest.mark.parametrize("x,y", [(None, None), ([[1]], torch.ones(1, 1)), (torch.ones(1, 1), [[1]])])
def test_input_types(x, y):
    with pytest.raises(TypeError, match="torch.Tensor"):
        _validation.validate_inputs(x, y)


@pytest.mark.parametrize("x,y", [
    (torch.zeros(2), torch.zeros(2, 1)),
    (torch.zeros(2, 1), torch.zeros(2)),
    (torch.zeros(2, 1), torch.zeros(2, 2)),
    (torch.zeros(2, 0), torch.zeros(3, 0)),
])
def test_input_shapes(x, y):
    with pytest.raises(ValueError, match="matrices|feature count"):
        _validation.validate_inputs(x, y)


@pytest.mark.parametrize("shape_x,shape_y", [((0, 2**23), (0, 2**23)), ((2**31, 1), (0, 1)), ((0, 1), (2**31, 1)), ((50000, 1), (50000, 1))])
def test_shape_overflow_without_allocation(shape_x, shape_y):
    def tensor(shape):
        obj = MagicMock(spec=torch.Tensor)
        obj.ndim, obj.shape = 2, shape
        return obj
    with pytest.raises(ValueError, match="feature count|too many pairs"):
        _validation.validate_inputs(tensor(shape_x), tensor(shape_y))


@pytest.mark.parametrize("dtype_x,dtype_y", [(torch.float64, torch.float32), (torch.float32, torch.float16)])
def test_dtype(dtype_x, dtype_y):
    with pytest.raises(TypeError, match="FP32"):
        _validation.validate_inputs(torch.zeros(1, 1, dtype=dtype_x), torch.zeros(1, 1, dtype=dtype_y))


@pytest.mark.parametrize("which", ["X", "Y"])
def test_sparse_rejected(which):
    x, y = torch.ones(1, 2), torch.ones(1, 2)
    if which == "X":
        x = x.to_sparse()
    else:
        y = y.to_sparse()
    with pytest.raises(TypeError, match="dense strided"):
        _validation.validate_inputs(x, y)


def test_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA tensors"):
        _validation.validate_inputs(torch.ones(1, 2), None)
    with pytest.raises(ValueError, match="CUDA tensors"):
        _validation.validate_inputs(fake_cuda([[1, 2]]), torch.ones(1, 2))


def test_devices(capability):
    class OtherDevice(DeviceTensor):
        @property
        def device(self):
            return torch.device("cuda:1")
    with pytest.raises(ValueError, match="same CUDA device"):
        _validation.validate_inputs(fake_cuda([[1]]), torch.ones(1, 1).as_subclass(OtherDevice))


@pytest.mark.parametrize("which", ["X", "Y"])
def test_gradients(which, capability):
    x, y = fake_cuda([[1]]), fake_cuda([[2]])
    (x if which == "X" else y).requires_grad_(True)
    with pytest.raises(ValueError, match="autograd"):
        _validation.validate_inputs(x, y)


@pytest.mark.parametrize("x,y", [([[float("nan")]], [[0]]), ([[0]], [[float("inf")]]), ([[-float("inf")]], [[0]])])
def test_finiteness(x, y, capability):
    with pytest.raises(ValueError, match="finite coordinates"):
        _validation.validate_inputs(fake_cuda(x), fake_cuda(y))


def test_old_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(RuntimeError, match="SM80"):
        _validation.validate_inputs(fake_cuda([[1]]), None)


def test_valid_inputs(capability):
    x = fake_cuda([[1, 2], [3, 4]]).t()
    assert not x.is_contiguous()
    a, b = _validation.validate_inputs(x, None)
    assert a.is_contiguous() and b.is_contiguous()
    assert torch.equal(a, x)
    _validation.validate_inputs(fake_cuda(torch.empty(0, 2)), fake_cuda([[1, 2]]))
    negative = x._neg_view()
    a, b = _validation.validate_inputs(negative, x)
    assert not a.is_neg() and torch.equal(a, -x)


@pytest.fixture
def dispatched(monkeypatch):
    """Stub native module entry points for dispatch tests."""
    calls = []
    def paper(x, y, *options):
        calls.append(("paper", options))
        d = (x.double()[:, None] - y.double()[None]).square().sum(-1).float()
        return d, torch.zeros_like(d, dtype=torch.uint8)
    def controlled(x, y, *options):
        calls.append(("controlled", options))
        d, mask = paper(x, y)
        return d, mask, torch.ones_like(d, dtype=torch.float64)
    def direct(x, y, precision):
        calls.append(("direct", precision))
        dtype = torch.float64 if precision == 64 else torch.float32
        return (x.to(dtype)[:, None] - y.to(dtype)[None]).square().sum(-1)
    def load(name):
        return SimpleNamespace(distance=paper if name == "paper" else controlled, direct=direct)
    monkeypatch.setattr(_api, "load_backend", load)
    monkeypatch.setattr(_api, "validate_inputs", lambda x, y: (x, x if y is None else y))
    return calls


@pytest.mark.parametrize("info", [False, True])
def test_paper_dispatch(dispatched, info):
    x = torch.tensor([[0., 0.], [3., 4.]])
    result = euclimix.pairwise_squared_distances(x, precision="bf16", correction="same", rho=10, return_info=info)
    if info:
        assert result.model == "original_empirical_test"
        assert result.absolute_bound is None
        assert result.fallback_count == 0 and result.fallback_fraction == 0
        assert result.fallback_mask.dtype == torch.bool
        result = result.distances
    assert result.tolist() == [[0, 25], [25, 0]]
    assert dispatched == [("paper", (1, 0, 10.0))]


@pytest.mark.parametrize("info,backend", [(False, "cublas"), (True, "cublas"), (True, "sequential")])
def test_controlled_dispatch(dispatched, info, backend):
    result = euclimix.controlled_squared_distances(torch.zeros(2, 3), torch.ones(1, 3), precision="bf16", correction="fp32", storage="low", test="original", backend=backend, return_info=info)
    assert dispatched[0] == ("controlled", (1, 0, int(backend == "sequential"), 1, 32, 32, 5.0, -1.0))
    if info:
        assert result.absolute_bound.dtype == torch.float64
        assert result.model == ("conditional_library_model" if backend == "cublas" else "sequential_fma_model")
        result = result.distances
    assert result.shape == (2, 1) and torch.all(result == 3)


@pytest.mark.parametrize("function", [euclimix.pairwise_squared_distances, euclimix.controlled_squared_distances])
def test_info_type(function):
    with pytest.raises(TypeError, match="return_info"):
        function(None, return_info=1)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_direct_dispatch(dispatched, precision):
    result = euclimix.direct_squared_distances(torch.tensor([[0., 0.]]), torch.tensor([[3., 4.]]), precision=precision)
    assert result.item() == 25
    assert result.dtype == (torch.float64 if precision == "fp64" else torch.float32)
    assert dispatched == [("direct", int(precision[2:]))]


def test_euclidean_dispatch(dispatched):
    d = euclimix.pairwise_distances(torch.tensor([[0., 0.]]), torch.tensor([[3., 4.]]))
    assert d.item() == 5


def test_empty_diagnostics():
    info = euclimix.DistanceResult(torch.empty(0, 2), torch.empty(0, 2, dtype=torch.bool), None, "test")
    assert info.fallback_fraction == 0
    info = euclimix.DistanceResult(torch.zeros(2, 2), torch.tensor([[True, False], [True, False]]), None, "test")
    assert info.fallback_count == 2 and info.fallback_fraction == 0.5


@pytest.fixture
def loader(monkeypatch):
    import torch.utils.cpp_extension as ext
    _extension.load_backend.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ext, "CUDA_HOME", "/fake/cuda")
    yield ext
    _extension.load_backend.cache_clear()


def test_invalid_backend(loader):
    with pytest.raises(ValueError, match="backend"):
        _extension.load_backend("bad")


def test_missing_cuda(loader, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _extension.load_backend("paper")


def test_missing_toolkit(loader, monkeypatch):
    monkeypatch.setattr(loader, "CUDA_HOME", None)
    with pytest.raises(RuntimeError, match="toolkit not found"):
        _extension.load_backend("paper")


def test_build_modules_and_cache(loader, monkeypatch):
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return kwargs["name"]
    monkeypatch.setattr(loader, "load", load)
    monkeypatch.setenv("EUCLIMIX_BUILD_VERBOSE", "1")
    assert euclimix.build_cuda() == ("euclimix_paper_cuda", "euclimix_controlled_cuda")
    assert euclimix.build_cuda() == ("euclimix_paper_cuda", "euclimix_controlled_cuda")
    assert len(calls) == 2
    assert "--use_fast_math" in calls[0]["extra_cuda_cflags"]
    assert "--ftz=false" in calls[1]["extra_cuda_cflags"]
    assert "--fmad=false" in calls[1]["extra_cuda_cflags"]
    assert "--use_fast_math" not in calls[1]["extra_cuda_cflags"]
    for call in calls:
        assert call["verbose"]
        assert call["extra_ldflags"] == ["-lcublas"]
        assert all(Path(p).is_file() for p in call["sources"])
        assert all(Path(p).is_dir() for p in call["extra_include_paths"])


@pytest.mark.parametrize("error", [OSError("missing compiler"), RuntimeError("bad compiler")])
def test_build_failure_preserves_cause(loader, monkeypatch, error):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(loader, "load", fail)
    with pytest.raises(RuntimeError, match="Could not build") as exc:
        _extension.load_backend("paper")
    assert exc.value.__cause__ is error

