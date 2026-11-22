"""Distance results and recomputation statistics."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistanceResult:
    """Distances and diagnostics for one evaluation.

    Parameters
    ----------
    distances : torch.Tensor of shape (n_samples_X, n_samples_Y)
        Squared distances, stored in FP32.
    fallback_mask : torch.Tensor of shape (n_samples_X, n_samples_Y)
        Boolean mask identifying entries recomputed by the direct formula.
    absolute_bound : torch.Tensor or None
        FP64 expanded-path bounds from the mask-based implementation. These
        apply before correction and are conditional for the cuBLAS backend.
        None for ``pairwise_squared_distances``.
    model : str
        Description of the arithmetic assumptions.

    Notes
    -----
    Reading ``fallback_count`` or ``fallback_fraction`` synchronizes with
    the GPU. The tensors themselves remain on the input device.
    """

    distances: torch.Tensor
    fallback_mask: torch.Tensor
    absolute_bound: torch.Tensor | None
    model: str

    @property
    def fallback_count(self) -> int:
        """Number of entries recomputed by the direct formula."""
        return int(self.fallback_mask.sum().item())

    @property
    def fallback_fraction(self) -> float:
        """Fraction of recomputed entries, or zero for an empty result."""
        size = self.fallback_mask.numel()
        return self.fallback_count / size if size else 0.0

