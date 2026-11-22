// Mixed-precision distances using the Original reliability test.
#include "common.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace euclimix {
namespace {
constexpr int threads = 256;
int blocks(int64_t count) { return static_cast<int>((count + threads - 1) / threads); }

__global__ void find_nonfinite(const float* x, int64_t nx, const float* y, int64_t ny, int* invalid) {
    const int64_t size = nx > ny ? nx : ny;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    bool bad = false;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride) {
        if (i < nx) bad |= !isfinite(x[i]);
        if (i < ny) bad |= !isfinite(y[i]);
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0) atomicExch(invalid, 1);
}

void check_finite(const torch::Tensor& x, const torch::Tensor& y) {
    const int64_t size = std::max(x.numel(), y.numel());
    if (size == 0) return;
    const int64_t max_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
    const int grid = static_cast<int>(std::min((size + threads - 1) / threads, max_blocks));
    auto invalid = torch::zeros({}, x.options().dtype(torch::kInt32));
    find_nonfinite<<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), x.numel(), y.data_ptr<float>(), y.numel(), invalid.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    TORCH_CHECK_VALUE(invalid.item<int>() == 0, "X and Y must contain only finite coordinates");
}

template<class T> __global__ void norms(const T* x, T* out, int n, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0;
    for (int k = 0; k < r; ++k) {
        float value = static_cast<float>(x[int64_t(i) * r + k]);
        sum = fmaf(value, value, sum);
    }
    out[i] = T(sum);
}

template<class T> __device__ float same_distance(const T* x, const T* y, int r);

template<> __device__ float same_distance<c10::Half>(const c10::Half* x, const c10::Half* y, int r) {
    __half sum = __float2half(0.f);
    for (int k = 0; k < r; ++k) {
        __half a = __float2half(static_cast<float>(x[k]));
        __half b = __float2half(static_cast<float>(y[k]));
        __half delta = __hsub(a, b);
        sum = __hadd(sum, __hmul(delta, delta));
    }
    return __half2float(sum);
}

template<> __device__ float same_distance<c10::BFloat16>(const c10::BFloat16* x, const c10::BFloat16* y, int r) {
    __nv_bfloat16 sum = __float2bfloat16(0.f);
    for (int k = 0; k < r; ++k) {
        __nv_bfloat16 a = __float2bfloat16(static_cast<float>(x[k]));
        __nv_bfloat16 b = __float2bfloat16(static_cast<float>(y[k]));
        __nv_bfloat16 delta = __hsub(a, b);
        sum = __hadd(sum, __hmul(delta, delta));
    }
    return __bfloat162float(sum);
}

template<> __device__ float same_distance<float>(const float* x, const float* y, int r) {
    float sum = 0;
    for (int k = 0; k < r; ++k) { float d = x[k] - y[k]; sum += d * d; }
    return sum;
}

template<class T> __device__ float corrected_distance(
    const float* x, const float* y, const T* xq, const T* yq, int r, int correction) {
    if (correction == 0) return same_distance(xq, yq, r);
    if (correction == 64) {
        double sum = 0;
        for (int k = 0; k < r; ++k) { double d = double(x[k]) - double(y[k]); sum += d * d; }
        return static_cast<float>(sum);
    }
    float sum = 0;
    for (int k = 0; k < r; ++k) { float d = x[k] - y[k]; sum += d * d; }
    return sum;
}

template<class T> __global__ void finish_distances(
    const T* cross, const T* xn, const T* yn, const float* x, const float* y,
    const T* xq, const T* yq, float* d, uint8_t* mask, int64_t total, int m, int r,
    int precision, int correction, float rho, bool euclidean) {
    const int64_t z = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (z >= total) return;
    const int64_t row = z / m, column = z % m;
    const float p = static_cast<float>(xn[row]);
    const float q = static_cast<float>(yn[column]);
    const float g = static_cast<float>(cross[z]);
    const float raw = p - 2.f * g + q;
    float value = fmaxf(raw, 0.f);
    // Format-dependent empirical coefficients for the Original test.
    float coefficient;
    if (precision == 2 && correction == 64) coefficient = 2.f * (r * 0x1p-23f);
    else coefficient = (2 * r + 4) * (precision == 0 ? 0x1p-11f : precision == 1 ? 0x1p-7f : 0x1p-23f);
    const float threshold = rho * coefficient * (p + q);
    const bool reject = !isfinite(raw) || !isfinite(threshold) || value <= threshold;
    if (reject) {
        const int64_t i = row * r, j = column * r;
        value = corrected_distance(x + i, y + j, xq + i, yq + j, r, correction);
    }
    if (mask) mask[z] = reject;
    if (euclidean) {
        // Match the standalone FP32 square root, including subnormal inputs.
        asm("sqrt.rn.f32 %0, %1;" : "=f"(value) : "f"(value));
    }
    d[z] = value;
}

template<class T> std::vector<torch::Tensor> evaluate(
    const torch::Tensor& x, const torch::Tensor& y, int precision, int correction, double rho,
    bool euclidean) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    int n = x.size(0), m = y.size(0), r = x.size(1);
    int64_t total = int64_t(n) * m;
    auto dtype = precision == 0 ? torch::kFloat16 : precision == 1 ? torch::kBFloat16 : torch::kFloat32;
    auto xq = x.to(dtype), yq = y.to(dtype);
    auto xn = torch::empty({n}, x.options().dtype(dtype));
    auto yn = torch::empty({m}, x.options().dtype(dtype));
    auto d = torch::empty({n, m}, x.options());
    auto mask = euclidean ? torch::Tensor() : torch::empty({n, m}, x.options().dtype(torch::kUInt8));
    if (!total) return {d, mask};
    norms<T><<<blocks(n), threads, 0, stream>>>(xq.data_ptr<T>(), xn.data_ptr<T>(), n, r);
    norms<T><<<blocks(m), threads, 0, stream>>>(yq.data_ptr<T>(), yn.data_ptr<T>(), m, r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    torch::Tensor cross;
    if (precision == 2) cross = torch::mm(x, y.t());
    else {
        cross = torch::empty({n, m}, x.options().dtype(dtype));
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        detail::BlasState saved(handle, stream);
        EUCLIMIX_BLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        EUCLIMIX_BLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        float alpha = 1, beta = 0;
        auto type = precision == 0 ? CUDA_R_16F : CUDA_R_16BF;
        EUCLIMIX_BLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, r,
            &alpha, yq.data_ptr(), type, r, xq.data_ptr(), type, r, &beta,
            cross.data_ptr(), type, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    finish_distances<T><<<blocks(total), threads, 0, stream>>>(
        cross.data_ptr<T>(), xn.data_ptr<T>(), yn.data_ptr<T>(), x.data_ptr<float>(), y.data_ptr<float>(),
        xq.data_ptr<T>(), yq.data_ptr<T>(), d.data_ptr<float>(), euclidean ? nullptr : mask.data_ptr<uint8_t>(),
        total, m, r, precision, correction, float(rho), euclidean);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d, mask};
}
std::vector<torch::Tensor> compute_distances(const torch::Tensor& x, const torch::Tensor& y,
                                            int precision, int correction, double rho, bool euclidean) {
    detail::check_input_metadata(x, y, precision == 1);
    TORCH_CHECK(precision >= 0 && precision <= 2, "precision must be 0, 1 or 2");
    TORCH_CHECK(correction == 0 || correction == 32 || correction == 64, "invalid correction precision");
    TORCH_CHECK(std::isfinite(rho) && rho > 1 && rho <= 1e6, "rho must be finite in (1, 1e6]");
    c10::cuda::CUDAGuard guard(x.device());
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    detail::record_inputs(x, y);
    check_finite(x, y);
    if (precision == 0) return evaluate<c10::Half>(x, y, precision, correction, rho, euclidean);
    if (precision == 1) return evaluate<c10::BFloat16>(x, y, precision, correction, rho, euclidean);
    return evaluate<float>(x, y, precision, correction, rho, euclidean);
}
}  // namespace

std::vector<torch::Tensor> paper_distance(torch::Tensor x, torch::Tensor y, int precision, int correction, double rho) {
    return compute_distances(x, y, precision, correction, rho, false);
}

torch::Tensor paper_euclidean_distance(torch::Tensor x, torch::Tensor y, int precision, int correction, double rho) {
    return compute_distances(x, y, precision, correction, rho, true)[0];
}
}  // namespace euclimix

