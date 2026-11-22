// Checks cuBLAS modes and stream on the same handle before and after a call.
#include "common.h"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

template<class Function> void check_modes(Function evaluate) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto current = c10::cuda::getCurrentCUDAStream();
    euclimix::detail::BlasState restore(handle, current);
    cublasMath_t before_math;
    cudaStream_t before_stream;
    EUCLIMIX_BLAS(cublasGetMathMode(handle, &before_math));
    EUCLIMIX_BLAS(cublasGetStream(handle, &before_stream));
    // Check restoration after GEMM temporarily uses host scalars.
    EUCLIMIX_BLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    auto result = evaluate();
    TORCH_CHECK(result.distances.template item<float>() == 25.f, "wrong distance");
    cublasMath_t after_math;
    cublasPointerMode_t after_pointer;
    cudaStream_t after_stream;
    EUCLIMIX_BLAS(cublasGetMathMode(handle, &after_math));
    EUCLIMIX_BLAS(cublasGetPointerMode(handle, &after_pointer));
    EUCLIMIX_BLAS(cublasGetStream(handle, &after_stream));
    TORCH_CHECK(after_math == before_math, "cuBLAS math mode was not restored");
    TORCH_CHECK(after_pointer == CUBLAS_POINTER_MODE_DEVICE, "cuBLAS pointer mode was not restored");
    TORCH_CHECK(after_stream == before_stream, "cuBLAS stream was changed");
}

int main() {
    c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto x = torch::zeros({1, 2}, options);
    auto y = torch::arange(3, 5, options).reshape({1, 2});
    const bool supports_bf16 = at::cuda::getCurrentDeviceProperties()->major >= 8;
    for (auto precision : {euclimix::InputPrecision::FP16, euclimix::InputPrecision::BF16}) {
        if (precision == euclimix::InputPrecision::BF16 && !supports_bf16) continue;
        euclimix::PaperOptions paper;
        paper.precision = precision;
        check_modes([&] { return euclimix::paper_squared_distances(x, y, paper); });
        euclimix::ControlledOptions controlled;
        controlled.precision = precision;
        check_modes([&] { return euclimix::controlled_squared_distances(x, y, controlled); });
    }
    std::cout << "euclimix cuBLAS state preservation OK\n";
}
