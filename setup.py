"""Build a source installation or a precompiled Linux CUDA wheel."""

import os
from pathlib import Path
import platform
import runpy
import sys

from setuptools import setup


def cuda_wheel_options():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    config = runpy.run_path(str(Path(__file__).resolve().parent / "src/euclimix/_build_config.py"))
    torch_version = config["WHEEL_TORCH_VERSION"]
    cuda_version = config["WHEEL_CUDA_VERSION"]
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("The precompiled CUDA wheel targets Linux x86_64.")
    if torch.__version__.split("+", 1)[0] != torch_version or torch.version.cuda != cuda_version:
        raise RuntimeError(f"Build the CUDA wheel with torch=={torch_version} and CUDA {cuda_version}.")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0"
    source_dir = Path("src/euclimix/csrc")
    extensions = [
        CUDAExtension(
            f"euclimix._{backend}_cuda",
            sources=[str(source_dir / f"{backend}_bindings.cpp"), str(source_dir / f"{backend}.cu")],
            include_dirs=[str((source_dir / "include").resolve())],
            libraries=["cublas"],
            runtime_library_dirs=["$ORIGIN/../torch/lib", "$ORIGIN/../nvidia/cu13/lib"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": list(flags)},
        )
        for backend, flags in config["CUDA_FLAGS"].items()
    ]
    return {
        "install_requires": [f"torch=={torch_version}"],
        "ext_modules": extensions,
        "cmdclass": {"build_ext": BuildExtension},
    }


if os.environ.get("EUCLIMIX_BUILD_CUDA_WHEEL") == "1":
    setup(**cuda_wheel_options())
else:
    setup(install_requires=["torch>=2.11,<3", "ninja>=1.11"])
