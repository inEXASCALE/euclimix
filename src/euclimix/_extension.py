"""CUDA extension loading."""

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
import os
from pathlib import Path

import torch

from ._build_config import CUDA_FLAGS, WHEEL_CUDA_VERSION, WHEEL_TORCH_VERSION


@lru_cache(maxsize=2)
def load_backend(backend):
    if backend not in ("paper", "controlled"):
        raise ValueError("backend must be 'paper' or 'controlled'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and install CUDA-enabled PyTorch.")

    module_name = f"euclimix._{backend}_cuda"
    if find_spec(module_name) is not None:
        if (
            torch.__version__.split("+", 1)[0] != WHEEL_TORCH_VERSION
            or torch.version.cuda != WHEEL_CUDA_VERSION
        ):
            raise RuntimeError(
                f"The precompiled euclimix wheel requires torch=={WHEEL_TORCH_VERSION} "
                f"with CUDA {WHEEL_CUDA_VERSION}. Found torch {torch.__version__} "
                f"with CUDA {torch.version.cuda}."
            )
        try:
            return import_module(module_name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"Could not load the precompiled CUDA module {module_name}: {exc}") from exc

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; set CUDA_HOME and put nvcc on PATH")
    root = Path(__file__).resolve().parent / "csrc"
    try:
        return load(
            name=f"euclimix_{backend}_cuda",
            sources=[str(root / f"{backend}_bindings.cpp"), str(root / f"{backend}.cu")],
            extra_include_paths=[str(root / "include")],
            extra_cflags=["-O3"], extra_cuda_cflags=list(CUDA_FLAGS[backend]),
            extra_ldflags=["-lcublas"],
            verbose=os.environ.get("EUCLIMIX_BUILD_VERBOSE") == "1",
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not build euclimix {backend!r}. Check nvcc, the C++ compiler, "
            "ninja, CUDA/PyTorch versions and TORCH_CUDA_ARCH_LIST. "
            "Set EUCLIMIX_BUILD_VERBOSE=1 for the compiler log."
        ) from exc


def build_cuda():
    """Load both CUDA extensions, compiling them for a source installation.

    Returns
    -------
    modules : tuple of modules
        Original-test and controlled CUDA extension modules.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or the extensions could not be loaded.

    Notes
    -----
    Precompiled wheels load both modules directly. Source installations use
    PyTorch's extension cache and build lock.
    """
    return load_backend("paper"), load_backend("controlled")

