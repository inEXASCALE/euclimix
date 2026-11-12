"""Small CUDA example, including a direct FP64 reference."""

import torch

from euclimix import controlled_squared_distances, direct_squared_distances, pairwise_squared_distances

X = torch.tensor([[0., 0.], [3., 4.]], device="cuda")
index_list = pairwise_squared_distances(X, return_info=True)
extended = controlled_squared_distances(X, return_info=True)
reference = direct_squared_distances(X)
print("Squared distances:", index_list.distances)
print("Extended squared distances:", extended.distances)
print("Direct FP64 reference:", reference)
print("Recomputation fraction:", index_list.fallback_fraction)

