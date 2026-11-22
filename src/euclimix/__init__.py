"""Mixed-precision CUDA distances with explicit arithmetic and correction modes."""

from ._api import (
    controlled_squared_distances,
    direct_squared_distances,
    pairwise_distances,
    pairwise_squared_distances,
)
from ._extension import build_cuda
from ._result import DistanceResult

__version__ = "0.1.2"
__all__ = [
    "DistanceResult", "build_cuda", "controlled_squared_distances",
    "direct_squared_distances", "pairwise_distances", "pairwise_squared_distances",
]

