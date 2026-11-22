"""Shared CUDA compiler settings and precompiled wheel requirements."""

WHEEL_TORCH_VERSION = "2.15.0"
WHEEL_CUDA_VERSION = "13.2"
CUDA_FLAGS = {
    "paper": ("-O3", "-lineinfo", "--use_fast_math"),
    "controlled": ("-O3", "-lineinfo", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"),
}
