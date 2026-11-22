"""Check the platform tag, CUDA machine code and build paths of a wheel."""

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


BUILD_PATH = re.compile(rb"/(?:release|home|Users|tmp|__w|__t|github/workspace|opt/rh)/[A-Za-z0-9._+/-]+")
CUDA_ARCHITECTURE = re.compile(r"\.sm_(\d+)\.cubin$")


def check_build_paths(path):
    leak = BUILD_PATH.search(path.read_bytes())
    if leak is not None:
        raise RuntimeError(f"{path.name} contains an absolute build path: {leak[0].decode()}")


def check(path):
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tag = f"{abi}-{abi}-manylinux_2_28_x86_64"
    with ZipFile(path) as wheel, TemporaryDirectory() as directory:
        names = wheel.namelist()
        metadata = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        if f"Tag: {tag}" not in wheel.read(metadata).decode().splitlines():
            raise RuntimeError(f"{path.name} is missing the expected {tag} wheel tag.")
        extensions = {
            PurePosixPath(name).name.split(".")[0]: name
            for name in names
            if name.startswith("euclimix/") and name.endswith(".so")
        }
        if set(extensions) != {"_paper_cuda", "_controlled_cuda"}:
            raise RuntimeError(f"Expected both CUDA extensions in {path.name}.")
        for module, name in extensions.items():
            module_dir = Path(directory) / module
            module_dir.mkdir()
            extracted = module_dir / PurePosixPath(name).name
            extracted.write_bytes(wheel.read(name))
            check_build_paths(extracted)
            listing = subprocess.check_output(
                ["cuobjdump", "--extract-elf", "all", extracted.name], cwd=module_dir, text=True,
            )
            architectures = set()
            for cubin in module_dir.glob("*.cubin"):
                if match := CUDA_ARCHITECTURE.search(cubin.name):
                    architectures.add(match[1])
                check_build_paths(cubin)
            if not {"75", "80"} <= architectures:
                raise RuntimeError(f"{module} is missing SM75 or SM80 machine code.\n{listing}")
            print(f"{path.name} {module} passed architecture and build-path checks for SM{' and SM'.join(sorted(architectures))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_cuda_wheel.py WHEEL")
    check(Path(sys.argv[1]))
