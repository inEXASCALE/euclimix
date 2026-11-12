#include "euclimix/distance.h"
#include <iostream>

int main() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto x = torch::zeros({1, 2}, options);
    auto y = torch::arange(3, 5, options).reshape({1, 2});
    auto result = euclimix::paper_squared_distances(x, y);
    TORCH_CHECK(result.distances.item<float>() == 25.f, "unexpected squared distance");
    euclimix::ControlledOptions controlled_options;
    controlled_options.backend = euclimix::CrossProduct::Sequential;
    auto controlled = euclimix::controlled_squared_distances(x, y, controlled_options);
    TORCH_CHECK(controlled.distances.item<float>() == 25.f, "unexpected controlled distance");
    TORCH_CHECK(euclimix::direct_squared_distances(x, y).item<double>() == 25., "unexpected direct distance");
    std::cout << "euclimix native CUDA API OK\n";
}

