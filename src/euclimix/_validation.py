"""Validation shared by the public interfaces."""

import math
from numbers import Real

import torch

FORMATS = {"fp16": 0, "bf16": 1, "fp32": 2}
CORRECTIONS = {"same": 0, "fp32": 32, "fp64": 64}
STORAGE = {"low": 0, "fp32": 1}
TESTS = {"original": 1, "extended": 2}
BACKENDS = {"cublas": 0, "sequential": 1}


def choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}; got {value!r}")
    return choices[value]


def validate_rho(rho):
    if isinstance(rho, bool) or not isinstance(rho, Real):
        raise TypeError("rho must be a real number")
    # Check the range before float conversion: arbitrarily large Python ints
    # are Real numbers but math.isfinite would raise OverflowError on them.
    if not 1 < rho <= 1e6 or not math.isfinite(rho):
        raise ValueError("rho must be finite and in (1, 1e6]")
    return float(rho)


def validate_inputs(X, Y, *, bf16=False, check_finite=True):
    Y = X if Y is None else Y
    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("X and Y must be torch.Tensor instances")
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be matrices with the same number of features")
    if not 0 < X.shape[1] < 2**23:
        raise ValueError("the feature count must be in [1, 2**23)")
    if X.shape[0] > 2**31 - 1 or Y.shape[0] > 2**31 - 1 or X.shape[0] * Y.shape[0] > 2**31 - 1:
        raise ValueError("too many pairs; split the computation into smaller batches")
    if X.dtype != torch.float32 or Y.dtype != torch.float32:
        raise TypeError("X and Y must contain the original FP32 coordinates")
    if X.layout != torch.strided or Y.layout != torch.strided:
        raise TypeError("dense strided tensors are required")
    if not X.is_cuda or not Y.is_cuda:
        raise ValueError("CUDA tensors are required; there is no implicit CPU fallback")
    if X.device != Y.device:
        raise ValueError("X and Y must be on the same CUDA device")
    if X.requires_grad or Y.requires_grad:
        raise ValueError("autograd is not supported; explicitly detach the inputs")
    if check_finite:
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("X and Y must contain only finite coordinates")
    capability = torch.cuda.get_device_capability(X.device)
    if capability < (7, 5):
        raise RuntimeError("euclimix requires an NVIDIA Turing or newer GPU (SM75+)")
    if bf16 and capability < (8, 0):
        raise RuntimeError("BF16 requires an NVIDIA Ampere or newer GPU (SM80+)")
    return X.resolve_neg().contiguous(), Y.resolve_neg().contiguous()

