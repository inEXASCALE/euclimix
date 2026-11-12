#pragma once

#include "euclimix/distance.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <cublas_v2.h>
#include <climits>
#include <cmath>

#define EUCLIMIX_BLAS(call) TORCH_CHECK((call) == CUBLAS_STATUS_SUCCESS, "cuBLAS failure: ", #call)

namespace euclimix::detail {

inline void record_inputs(const torch::Tensor& x, const torch::Tensor& y) {
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(y.storage().data_ptr(), stream);
}

inline void check_inputs(const torch::Tensor& x, const torch::Tensor& y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && x.device() == y.device(),
                "X and Y must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32,
                "original FP32 coordinates are required");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
                "expected matrices with matching feature counts");
    TORCH_CHECK(x.layout() == c10::kStrided && y.layout() == c10::kStrided,
                "dense strided coordinates are required");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "contiguous coordinates are required");
    TORCH_CHECK(!x.is_neg() && !y.is_neg(), "resolve negative views before native calls");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) < (1 << 23), "invalid feature count");
    TORCH_CHECK(x.size(0) <= INT_MAX && y.size(0) <= INT_MAX &&
                x.size(0) * y.size(0) <= INT_MAX, "too many pairs; batch the inputs");
    TORCH_CHECK(!x.requires_grad() && !y.requires_grad(), "autograd is unsupported; detach inputs");
    TORCH_CHECK(torch::isfinite(x).all().item<bool>() && torch::isfinite(y).all().item<bool>(),
                "coordinates must be finite");
    TORCH_CHECK(at::cuda::getDeviceProperties(x.get_device())->major >= 8, "SM80+ GPU required");
}

// getCurrentCUDABlasHandle() already binds the current stream AND configures
// PyTorch's workspace. Never call cublasSetStream on the borrowed handle:
// even setting the same stream discards that workspace (cuBLAS API 2.4.7).
// Restore only the modes we change; destructors must not throw on unwinding.
struct BlasState {
    cublasHandle_t handle;
    cublasMath_t math;
    cublasPointerMode_t pointer;
    explicit BlasState(cublasHandle_t h, cudaStream_t expected_stream) : handle(h) {
        cudaStream_t stream;
        EUCLIMIX_BLAS(cublasGetStream(h, &stream));
        TORCH_CHECK(stream == expected_stream, "cuBLAS handle does not use the current CUDA stream");
        EUCLIMIX_BLAS(cublasGetMathMode(h, &math));
        EUCLIMIX_BLAS(cublasGetPointerMode(h, &pointer));
    }
    ~BlasState() {
        cublasSetMathMode(handle, math);
        cublasSetPointerMode(handle, pointer);
    }
};

}  // namespace euclimix::detail

