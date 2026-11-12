"""CUDA extension loading."""

from functools import lru_cache
import os
from pathlib import Path

import torch


@lru_cache(maxsize=2)
def load_backend(backend):
    if backend not in ("paper", "controlled"):
        raise ValueError("backend must be 'paper' or 'controlled'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; set CUDA_HOME and put nvcc on PATH")
    root = Path(__file__).resolve().parent / "csrc"
    flags = ["-O3", "-lineinfo"]
    if backend == "paper":
        flags += ["--use_fast_math"]
    else:
        flags += ["--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
    try:
        return load(
            name=f"euclimix_{backend}_cuda",
            sources=[str(root / f"{backend}_bindings.cpp"), str(root / f"{backend}.cu")],
            extra_include_paths=[str(root / "include")],
            extra_cflags=["-O3"], extra_cuda_cflags=flags,
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
    """Compile and load the CUDA extensions.

    Returns
    -------
    modules : tuple of modules
        Index-list and mask-based CUDA extension modules.

    Raises
    ------
    RuntimeError
        If CUDA, the toolkit, or a compatible compiler is unavailable.

    Notes
    -----
    Uses PyTorch's extension cache and build lock. Compilation is lazy;
    importing euclimix does not require nvcc or an attached GPU.
    """
    return load_backend("paper"), load_backend("controlled")

