// Index-list mixed-precision squared distances.
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace euclimix {
namespace {
constexpr int threads = 256;
int blocks(int64_t count) { return static_cast<int>((count + threads - 1) / threads); }

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

template<class T> __global__ void select_distances(
    const T* cross, const T* xn, const T* yn, float* d, uint8_t* mask,
    uint32_t* list, uint32_t* count, int64_t total, int m, int r,
    int precision, int correction, float rho) {
    int64_t z = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (z >= total) return;
    float p = static_cast<float>(xn[z / m]);
    float q = static_cast<float>(yn[z % m]);
    float g = static_cast<float>(cross[z]);
    float raw = p - 2.f * g + q;
    float value = fmaxf(raw, 0.f);
    // Format-dependent empirical coefficients for the Original test.
    float coefficient;
    if (precision == 2 && correction == 64) coefficient = 2.f * (r * 0x1p-23f);
    else coefficient = (2 * r + 4) * (precision == 0 ? 0x1p-11f : precision == 1 ? 0x1p-7f : 0x1p-23f);
    float threshold = rho * coefficient * (p + q);
    bool reject = !isfinite(raw) || !isfinite(threshold) || value <= threshold;
    d[z] = value;
    mask[z] = reject;
    if (reject) list[atomicAdd(count, 1u)] = static_cast<uint32_t>(z);
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

template<class T> __global__ void correct_distances(
    const float* x, const float* y, const T* xq, const T* yq, float* out,
    const uint32_t* list, const uint32_t* count, int m, int r, int correction) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *count) return;
    uint32_t z = list[t];
    int64_t i = int64_t(z / m) * r, j = int64_t(z % m) * r;
    if (correction == 0) out[z] = same_distance(xq + i, yq + j, r);
    else if (correction == 64) {
        double sum = 0;
        for (int k = 0; k < r; ++k) { double d = double(x[i+k]) - double(y[j+k]); sum += d * d; }
        out[z] = static_cast<float>(sum);
    } else {
        float sum = 0;
        for (int k = 0; k < r; ++k) { float d = x[i+k] - y[j+k]; sum += d * d; }
        out[z] = sum;
    }
}

template<class T> std::vector<torch::Tensor> evaluate(
    const torch::Tensor& x, const torch::Tensor& y, int precision, int correction, double rho) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    int n = x.size(0), m = y.size(0), r = x.size(1);
    int64_t total = int64_t(n) * m;
    auto dtype = precision == 0 ? torch::kFloat16 : precision == 1 ? torch::kBFloat16 : torch::kFloat32;
    auto xq = x.to(dtype), yq = y.to(dtype);
    auto xn = torch::empty({n}, x.options().dtype(dtype));
    auto yn = torch::empty({m}, x.options().dtype(dtype));
    auto d = torch::empty({n, m}, x.options());
    auto mask = torch::empty({n, m}, x.options().dtype(torch::kUInt8));
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
    auto list = torch::empty({total}, x.options().dtype(torch::kInt32));
    auto count = torch::zeros({1}, x.options().dtype(torch::kInt32));
    select_distances<T><<<blocks(total), threads, 0, stream>>>(
        cross.data_ptr<T>(), xn.data_ptr<T>(), yn.data_ptr<T>(), d.data_ptr<float>(),
        mask.data_ptr<uint8_t>(), reinterpret_cast<uint32_t*>(list.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(count.data_ptr<int32_t>()), total, m, r, precision, correction, float(rho));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    correct_distances<T><<<blocks(total), threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), xq.data_ptr<T>(), yq.data_ptr<T>(), d.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(list.data_ptr<int32_t>()), reinterpret_cast<uint32_t*>(count.data_ptr<int32_t>()), m, r, correction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d, mask};
}
}  // namespace

std::vector<torch::Tensor> paper_distance(torch::Tensor x, torch::Tensor y, int precision, int correction, double rho) {
    detail::check_inputs(x, y);
    TORCH_CHECK(precision >= 0 && precision <= 2, "precision must be 0, 1 or 2");
    TORCH_CHECK(correction == 0 || correction == 32 || correction == 64, "invalid correction precision");
    TORCH_CHECK(std::isfinite(rho) && rho > 1 && rho <= 1e6, "rho must be finite in (1, 1e6]");
    c10::cuda::CUDAGuard guard(x.device());
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    detail::record_inputs(x, y);
    if (precision == 0) return evaluate<c10::Half>(x, y, precision, correction, rho);
    if (precision == 1) return evaluate<c10::BFloat16>(x, y, precision, correction, rho);
    return evaluate<float>(x, y, precision, correction, rho);
}
}  // namespace euclimix

