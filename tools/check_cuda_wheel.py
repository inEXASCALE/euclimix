"""Check the platform tag and CUDA machine code of a built wheel."""

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


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
            extracted = Path(directory) / PurePosixPath(name).name
            extracted.write_bytes(wheel.read(name))
            listing = subprocess.check_output(["cuobjdump", "--list-elf", str(extracted)], text=True)
            architectures = set(re.findall(r"\.sm_(\d+)\.cubin", listing))
            if not {"75", "80"} <= architectures:
                raise RuntimeError(f"{module} is missing SM75 or SM80 machine code.\n{listing}")
            print(f"{path.name} {module} compiled for SM{' and SM'.join(sorted(architectures))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_cuda_wheel.py WHEEL")
    check(Path(sys.argv[1]))
