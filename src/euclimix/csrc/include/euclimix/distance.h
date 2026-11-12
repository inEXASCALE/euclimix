#pragma once

#include <torch/types.h>
#include <vector>

namespace euclimix {

// All coordinates are original, contiguous FP32 CUDA tensors. Results use
// the current PyTorch CUDA stream. Native entry points validate independently
// of Python. precision: 0=FP16, 1=BF16, 2=FP32; correction: 0=same, 32, 64.
// Returns {FP32 squared distances, byte rejection mask}.
std::vector<torch::Tensor> paper_distance(torch::Tensor x, torch::Tensor y,
                                        int precision, int correction, double rho);

// Mask-based implementation. storage: 0=low, 1=FP32;
// backend: 0=cuBLAS, 1=sequential; test: 1=Original, 2=Extended.
// output is 32 or 64. epsilon=-1 selects the conditional gamma_r
// assumption; a nonnegative coefficient cannot lower the norm error bound.
// Returns {distances, byte rejection mask, FP64 default-path absolute bound}.
std::vector<torch::Tensor> controlled_distance(
    torch::Tensor x, torch::Tensor y, int precision, int storage, int backend,
    int test, int correction, int output, double rho, double epsilon);

// Direct subtraction followed by sequential FP32/FP64 FMA. The output retains
// the specified precision. No input quantization is performed.
torch::Tensor direct_distance(torch::Tensor x, torch::Tensor y, int precision);

// Typed interface for C++ applications; integer entry points serve pybind11.
enum class InputPrecision { FP16 = 0, BF16 = 1, FP32 = 2 };
enum class Correction { Same = 0, FP32 = 32, FP64 = 64 };
enum class Storage { Low = 0, FP32 = 1 };
enum class ReliabilityTest { Original = 1, Extended = 2 };
enum class CrossProduct { CuBlas = 0, Sequential = 1 };
enum class OutputPrecision { FP32 = 32, FP64 = 64 };

struct PaperOptions {
    InputPrecision precision = InputPrecision::FP16;
    Correction correction = Correction::FP64;
    double rho = 5.0;
};

struct ControlledOptions {
    InputPrecision precision = InputPrecision::FP16;
    Correction correction = Correction::FP64;
    Storage storage = Storage::FP32;
    ReliabilityTest test = ReliabilityTest::Extended;
    CrossProduct backend = CrossProduct::CuBlas;
    OutputPrecision output = OutputPrecision::FP32;
    double rho = 5.0;
    double epsilon_a = -1.0;
};

struct DistanceResult {
    torch::Tensor distances;
    torch::Tensor fallback_mask;
    torch::Tensor absolute_bound;  // Undefined for the index-list implementation.
};

inline DistanceResult paper_squared_distances(
    torch::Tensor x, torch::Tensor y, const PaperOptions& options = {}) {
    auto result = paper_distance(x, y, static_cast<int>(options.precision),
                                 static_cast<int>(options.correction), options.rho);
    return {result[0], result[1], torch::Tensor()};
}

inline DistanceResult controlled_squared_distances(
    torch::Tensor x, torch::Tensor y, const ControlledOptions& options = {}) {
    auto result = controlled_distance(x, y, static_cast<int>(options.precision),
        static_cast<int>(options.storage), static_cast<int>(options.backend),
        static_cast<int>(options.test), static_cast<int>(options.correction),
        static_cast<int>(options.output), options.rho, options.epsilon_a);
    return {result[0], result[1], result[2]};
}

inline torch::Tensor direct_squared_distances(
    torch::Tensor x, torch::Tensor y, OutputPrecision precision = OutputPrecision::FP64) {
    return direct_distance(x, y, static_cast<int>(precision));
}

}  // namespace euclimix

