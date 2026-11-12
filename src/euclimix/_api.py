"""Euclidean and squared-distance interfaces."""

import torch

from ._extension import load_backend
from ._result import DistanceResult
from ._validation import (
    BACKENDS, CORRECTIONS, FORMATS, STORAGE, TESTS, choice, validate_inputs, validate_rho,
)


def pairwise_squared_distances(X, Y=None, *, precision="fp16", correction="fp64", rho=5.0, return_info=False):
    """Compute mixed-precision squared Euclidean distances.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples_X, n_features)
        Finite FP32 CUDA coordinates. Gradients are not supported.
    Y : torch.Tensor of shape (n_samples_Y, n_features), default=None
        Finite FP32 coordinates on the same device. If None, use X.
        Noncontiguous inputs are copied to contiguous storage.
    precision : {'fp16', 'bf16', 'fp32'}, default='fp16'
        Expanded-path input and stored intermediate format. FP16/BF16
        cross products use cuBLAS with FP32 compute; FP32 uses torch.mm.
    correction : {'fp32', 'fp64', 'same'}, default='fp64'
        Direct recomputation arithmetic. FP32/FP64 use the original FP32
        coordinates. 'same' uses quantized inputs and the selected low
        arithmetic, which can lose accuracy or overflow.
    rho : float, default=5.0
        Empirical threshold multiplier, finite and in (1, 1e6].
    return_info : bool, default=False
        If True, also return the per-call rejection mask and metadata.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples_X, n_samples_Y) or DistanceResult
        Nonnegative squared distances stored in FP32. If return_info is
        True, return a DistanceResult instead of a tensor.

    Notes
    -----
    Uses an index list and the Original empirical threshold. The FP32-to-FP32
    and FP32-to-FP64 paths use different threshold coefficients.
    FP64 correction is rounded to FP32 on output. Values outside the output
    range can overflow; 'same' can also lose accuracy or overflow on modest
    inputs.
    Native calls use the current CUDA stream; input validation synchronizes.
    """
    low = choice(precision, FORMATS, "precision")
    high = choice(correction, CORRECTIONS, "correction")
    rho = validate_rho(rho)
    if not isinstance(return_info, bool):
        raise TypeError("return_info must be a bool")
    X, Y = validate_inputs(X, Y)
    distances, mask = load_backend("paper").distance(X, Y, low, high, rho)
    if return_info:
        return DistanceResult(distances, mask.bool(), None, "original_empirical_test")
    return distances


def controlled_squared_distances(X, Y=None, *, precision="fp16", correction="fp64", storage="fp32", test="extended", backend="cublas", rho=5.0, return_info=False):
    """Compute squared distances with configurable storage and reliability tests.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples_X, n_features)
        Finite original FP32 CUDA coordinates without gradients.
    Y : torch.Tensor of shape (n_samples_Y, n_features), default=None
        Coordinates on the same device. If None, use X.
    precision : {'fp16', 'bf16'}, default='fp16'
        Format to which coordinates are quantized.
    correction : {'fp32', 'fp64'}, default='fp64'
        Direct recomputation arithmetic on the original FP32 inputs.
        The result is always returned in FP32.
    storage : {'low', 'fp32'}, default='fp32'
        Stored norm and cross-product format. 'low' uses precision.
    test : {'original', 'extended'}, default='extended'
        Original uses a format-dependent empirical coefficient. Extended uses
        conversion residuals, accumulation/storage errors and underflow.
    backend : {'cublas', 'sequential'}, default='cublas'
        Cross-product implementation. Both use FP32 accumulation. The
        cuBLAS error bound is conditional on an inner-product error
        assumption; sequential uses coordinate-order FP32 FMA.
    rho : float, default=5.0
        Threshold multiplier, finite and in (1, 1e6].
    return_info : bool, default=False
        If True, return distances, mask, absolute bounds and model label.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples_X, n_samples_Y) or DistanceResult
        Squared FP32 distances or the corresponding diagnostics.

    Notes
    -----
    Uses a rejection mask and strict floating-point compiler settings. The
    Extended relative-error tolerance is 1/(rho-1) for accepted distances,
    conditional on the inner-product error assumption. It does not cover
    output rounding after direct recomputation. Both tests compute bounds.
    """
    low = choice(precision, {"fp16": 0, "bf16": 1}, "precision")
    high = choice(correction, {"fp32": 32, "fp64": 64}, "correction")
    stored = choice(storage, STORAGE, "storage")
    criterion = choice(test, TESTS, "test")
    cross = choice(backend, BACKENDS, "backend")
    rho = validate_rho(rho)
    if not isinstance(return_info, bool):
        raise TypeError("return_info must be a bool")
    X, Y = validate_inputs(X, Y)
    distances, mask, bound = load_backend("controlled").distance(
        X, Y, low, stored, cross, criterion, high, 32, rho, -1.0,
    )
    if return_info:
        model = "conditional_library_model" if backend == "cublas" else "sequential_fma_model"
        return DistanceResult(distances, mask.bool(), bound, model)
    return distances


def direct_squared_distances(X, Y=None, *, precision="fp64"):
    """Evaluate the direct squared-distance formula on original coordinates.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples_X, n_features)
        Finite original FP32 CUDA coordinates without gradients.
    Y : torch.Tensor of shape (n_samples_Y, n_features), default=None
        Same-device FP32 coordinates. If None, use X.
    precision : {'fp32', 'fp64'}, default='fp64'
        Subtraction, FMA accumulation and output format. FP64 output is
        retained, making this suitable as a benchmark reference.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples_X, n_samples_Y)
        Squared distances in the selected precision.
    """
    high = choice(precision, {"fp32": 32, "fp64": 64}, "precision")
    X, Y = validate_inputs(X, Y)
    return load_backend("controlled").direct(X, Y, high)


def pairwise_distances(X, Y=None, *, precision="fp16", correction="fp64", rho=5.0):
    """Compute Euclidean distances by taking the square root of squared distances.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples_X, n_features)
        Finite original FP32 CUDA coordinates without gradients.
    Y : torch.Tensor of shape (n_samples_Y, n_features), default=None
        Same-device FP32 coordinates. If None, use X.
    precision : {'fp16', 'bf16', 'fp32'}, default='fp16'
        Default-path format; see pairwise_squared_distances.
    correction : {'fp32', 'fp64', 'same'}, default='fp64'
        Direct recomputation arithmetic; see pairwise_squared_distances.
    rho : float, default=5.0
        Empirical reliability threshold multiplier.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples_X, n_samples_Y)
        Euclidean distances in FP32, including square-root rounding.
    """
    return torch.sqrt(pairwise_squared_distances(
        X, Y, precision=precision, correction=correction, rho=rho,
    ))

