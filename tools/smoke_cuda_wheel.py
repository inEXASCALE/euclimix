"""Exercise an installed CUDA wheel on a machine with the driver only."""

from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys

import euclimix
import torch


def main():
    assert shutil.which("nvcc") is None, "nvcc is visible on PATH"
    assert not Path("/usr/local/cuda/bin/nvcc").exists(), "the system CUDA Toolkit is visible"
    assert not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH")
    jit_cache = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    assert not jit_cache.exists(), f"expected a fresh JIT cache directory: {jit_cache}"

    assert version("torch") == "2.15.0" and torch.version.cuda == "13.2"
    assert torch.cuda.is_available(), "A GPU with a compatible NVIDIA driver is required"
    major, minor = torch.cuda.get_device_capability()
    modules = euclimix.build_cuda()
    for backend, module in zip(("paper", "controlled"), modules):
        assert module.__name__ == f"euclimix._{backend}_cuda"
        extension = Path(module.__file__)
        assert extension.suffix == ".so" and extension.is_relative_to(Path(sys.prefix))

    with torch.inference_mode():
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]], device="cuda")
        distances = torch.tensor([[0.0, 5.0], [5.0, 0.0]], device="cuda")
        squared = distances.square()
        rejection_mask = torch.eye(2, dtype=torch.bool, device="cuda")
        precisions = ("fp16", "bf16") if major >= 8 else ("fp16",)
        for precision in precisions:
            options = {"precision": precision, "correction": "fp32"}
            assert torch.equal(euclimix.pairwise_distances(x, **options), distances)
            result = euclimix.pairwise_squared_distances(x, return_info=True, **options)
            assert torch.equal(result.distances, squared)
            assert torch.equal(result.fallback_mask, rejection_mask)
            assert torch.equal(euclimix.controlled_squared_distances(x, **options), squared)
        assert torch.equal(euclimix.direct_squared_distances(x), squared.double())

    assert "torch.utils.cpp_extension" not in sys.modules
    assert not jit_cache.exists()
    print(f"Python {sys.version.split()[0]} | torch {version('torch')} | CUDA {torch.version.cuda}")
    print(f"{torch.cuda.get_device_name()} (SM{major}{minor}) | {', '.join(precisions)}")
    print("Both precompiled CUDA backends and all four distance APIs passed without nvcc or JIT.")


if __name__ == "__main__":
    main()
