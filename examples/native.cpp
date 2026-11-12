// Include this header from an application linked with the CMake euclimix target.
#include <euclimix/distance.h>

euclimix::DistanceResult example(torch::Tensor x, torch::Tensor y) {
    // Original FP32 CUDA coordinates, contiguous on one device.
    // FP16 expanded path, Original criterion, direct FP64 correction.
    euclimix::PaperOptions options;
    options.precision = euclimix::InputPrecision::FP16;
    options.correction = euclimix::Correction::FP64;
    return euclimix::paper_squared_distances(x, y, options);
}

