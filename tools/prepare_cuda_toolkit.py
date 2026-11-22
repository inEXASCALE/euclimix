"""Assemble the CUDA 13.2 build tools for the PyPI PyTorch wheel."""

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_crt", "libnvvm", "cuda_cuobjdump")


def download(url, destination):
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--max-time", "180", "-o", str(destination), url],
        check=True,
    )


def prepare(prefix):
    torch = find_spec("torch")
    if torch is None or torch.origin is None:
        raise RuntimeError("Install the PyPI PyTorch wheel before preparing the CUDA build tools.")
    runtime = Path(torch.origin).resolve().parent.parent / "nvidia" / "cu13"
    cublas = runtime / "lib" / "libcublas.so.13"
    if not cublas.is_file() or not (runtime / "include" / "cublas_v2.h").is_file():
        raise RuntimeError("The NVIDIA cuBLAS libraries and headers from PyPI are required.")

    prefix.mkdir(parents=True)
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "component.tar.xz"
        manifest = Path(directory) / "manifest.json"
        download(f"{REDIST}/redistrib_13.2.2.json", manifest)
        components = json.loads(manifest.read_text())
        for name in COMPONENTS:
            component = components[name]["linux-x86_64"]
            download(f"{REDIST}/{component['relative_path']}", archive)
            with archive.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != component["sha256"]:
                raise RuntimeError(f"SHA-256 verification failed for {name}.")
            subprocess.run(
                ["tar", "-xf", str(archive), "-C", str(prefix), "--strip-components=1", "--no-same-owner"],
                check=True,
            )

    for header in (runtime / "include").iterdir():
        target = prefix / "include" / header.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(os.path.relpath(header, target.parent), target_is_directory=header.is_dir())
    library = prefix / "lib" / "libcublas.so"
    library.symlink_to(os.path.relpath(cublas, library.parent))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: prepare_cuda_toolkit.py CUDA_HOME")
    prepare(Path(sys.argv[1]).resolve())
