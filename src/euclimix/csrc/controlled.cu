// Mask-based mixed-precision distances with directed error bounds.
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace euclimix {
using torch::Tensor;
constexpr int THREADS = 256;
static int blocks(int64_t n) { return int((n + THREADS - 1) / THREADS); }
__device__ double add_up(double a, double b) { return __dadd_ru(a,b); }
__device__ double mul_up(double a, double b) { return __dmul_ru(a,b); }
__device__ double gamma_up(int n) {
  double nu = mul_up(double(n), 0x1p-24);
  return __ddiv_ru(nu, __dsub_rd(1.0,nu));
}
__device__ float rounded(float x, int low) {
  return low == 0 ? __half2float(__float2half_rn(x)) : __bfloat162float(__float2bfloat16_rn(x));
}
// All squared norms used to construct the bound use directed FP64 rounding.
// Columns: norm(original), norm(quantized), norm(actual conversion residual).
__global__ void norm_bounds(const float* x, const float* q, double* b, float* norm,
                                     int n, int r, int low, int storage) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  double sx=0, sq=0, se=0; float s=0;
  for (int k=0;k<r;++k) {
    int64_t z=int64_t(i)*r+k; double a=x[z], v=q[z];
    sx=add_up(sx,mul_up(a,a)); sq=add_up(sq,mul_up(v,v));
    double er=fmax(fabs(__dsub_ru(a,v)),fabs(__dsub_rd(a,v)));
    se=add_up(se,mul_up(er,er));
    s=__fmaf_rn(q[z],q[z],s);
  }
  b[int64_t(i)*3]=__dsqrt_ru(sx);
  b[int64_t(i)*3+1]=__dsqrt_ru(sq);
  b[int64_t(i)*3+2]=__dsqrt_ru(se);
  norm[i]=storage==0 ? rounded(s,low) : s;
}
__global__ void sequential_cross(const float* x,const float* c,float* g,int n,int m,int r,int low,int storage) {
  int64_t z=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(z>=int64_t(n)*m)return;
  int i=z/m,j=z%m;float s=0;
  for(int k=0;k<r;++k)s=__fmaf_rn(x[int64_t(i)*r+k],c[int64_t(j)*r+k],s);
  g[z]=storage==0?rounded(s,low):s;
}
__device__ double total_bound(const double* xb,const double* cb,int r,int low,int storage,double epsilon) {
  double ga=epsilon<0?gamma_up(r):fmax(epsilon,gamma_up(r));
  double us=storage==0?(low==0?0x1p-11:0x1p-8):0x1p-24;
  // A full subnormal spacing is used (more conservative than half an ulp).
  double etas=storage==0?(low==0?0x1p-24:0x1p-133):0x1p-149;
  double gw=gamma_up(2);
  double factor=mul_up(add_up(1.0,ga),add_up(1.0,us));
  double beta=__dsub_ru(mul_up(factor,add_up(1.0,gw)),1.0);
  double sq=add_up(mul_up(xb[1],xb[1]),mul_up(cb[1],cb[1]));
  double e=add_up(xb[2],cb[2]), norms=add_up(xb[0],cb[0]);
  double conversion=add_up(mul_up(mul_up(2.0,norms),e),mul_up(e,e));
  double den=__dsub_rd(1.0,mul_up(double(r),0x1p-24));
  double acc_absolute=__ddiv_ru(mul_up(double(r),0x1p-149),den);
  double stored_absolute=add_up(mul_up(add_up(1.0,us),acc_absolute),etas);
  double formation_absolute=__ddiv_ru(2*0x1p-149,__dsub_rd(1.0,2*0x1p-24));
  double arithmetic=add_up(mul_up(mul_up(2.0,beta),sq),
      add_up(mul_up(mul_up(4.0,add_up(1.0,gw)),stored_absolute),formation_absolute));
  return add_up(conversion,arithmetic);
}
template<class O> __global__ void select_distances(const float* cross,const float* pn,const float* cn,
  const double* pb,const double* cb,O* out,uint8_t* mask,double* bound,int n,int m,int r,
  int low,int storage,int threshold,double rho,double epsilon) {
  int64_t z=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(z>=int64_t(n)*m)return;
  int i=z/m,j=z%m;
  float raw=__fadd_rn(__fsub_rn(pn[i],__fmul_rn(2.f,cross[z])),cn[j]);
  float d=fmaxf(raw,0.f);
  double b=total_bound(pb+int64_t(i)*3,cb+int64_t(j)*3,r,low,storage,epsilon);
  bound[z]=b; bool fail=!isfinite(raw)||!isfinite(b);
  if(threshold==1){ // Original empirical coefficient; same explicit evaluation order.
    float u=low==0?0x1p-11f:0x1p-7f;
    float t=__fmul_rn(__fmul_rn(float(rho),__fmul_rn(float(2*r+4),u)),__fadd_rn(pn[i],cn[j]));
    fail=fail||d<=t;
  }else if(threshold==2){ fail=fail||double(d)<=mul_up(rho,b); }
  // Nonfinite estimates require direct recomputation.
  mask[z]=fail;out[z]=O(d);
}
template<class O,int CORRECTION> __global__ void recompute_distances(const float* x,const float* c,
  O* out,const uint8_t* mask,int n,int m,int r) {
  int64_t z=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(z>=int64_t(n)*m || (mask && !mask[z]))return;
  int i=z/m,j=z%m;
  if(CORRECTION==64){double s=0;for(int k=0;k<r;++k){double d=__dsub_rn(double(x[int64_t(i)*r+k]),double(c[int64_t(j)*r+k]));s=__fma_rn(d,d,s);}out[z]=O(s);}
  else {float s=0;for(int k=0;k<r;++k){float d=__fsub_rn(x[int64_t(i)*r+k],c[int64_t(j)*r+k]);s=__fmaf_rn(d,d,s);}out[z]=O(s);}
}
std::vector<Tensor> controlled_distance(Tensor x,Tensor c,int low,int storage,int backend,int threshold,
    int correction,int output,double rho,double epsilon) {
  detail::check_inputs(x,c);c10::cuda::CUDAGuard guard(x.device());
  detail::record_inputs(x,c);
  TORCH_CHECK((low==0||low==1)&&(storage==0||storage==1)&&(backend==0||backend==1),"invalid arithmetic mode");
  TORCH_CHECK(threshold>=1&&threshold<=2&&(correction==32||correction==64)&&(output==32||output==64),"invalid mode");
  TORCH_CHECK(std::isfinite(rho)&&rho>1&&rho<=1e6,"rho must be finite in (1,1e6]");
  TORCH_CHECK(epsilon==-1 || (std::isfinite(epsilon)&&epsilon>=0),"invalid epsilon");
  TORCH_CHECK(backend!=1||epsilon==-1,"sequential bound uses its derived gamma, not a user override");
  int n=x.size(0),m=c.size(0),r=x.size(1);auto stream=c10::cuda::getCurrentCUDAStream();
  if (n == 0 || m == 0) return {torch::empty({n,m},x.options().dtype(output==32?torch::kFloat32:torch::kFloat64)), torch::empty({n,m},x.options().dtype(torch::kUInt8)), torch::empty({n,m},x.options().dtype(torch::kFloat64))};
  auto type=low==0?torch::kFloat16:torch::kBFloat16;
  auto xl=x.to(type),cl=c.to(type);auto xq=xl.to(torch::kFloat32),cq=cl.to(torch::kFloat32);
  auto pn=torch::empty({n},x.options()),cn=torch::empty({m},x.options());
  auto pb=torch::empty({n,3},x.options().dtype(torch::kFloat64)),cb=torch::empty({m,3},x.options().dtype(torch::kFloat64));
  norm_bounds<<<blocks(n),THREADS,0,stream>>>(x.data_ptr<float>(),xq.data_ptr<float>(),pb.data_ptr<double>(),pn.data_ptr<float>(),n,r,low,storage);
  norm_bounds<<<blocks(m),THREADS,0,stream>>>(c.data_ptr<float>(),cq.data_ptr<float>(),cb.data_ptr<double>(),cn.data_ptr<float>(),m,r,low,storage);
  Tensor cross;
  if(backend==1){
    cross=torch::empty({n,m},x.options());
    sequential_cross<<<blocks(int64_t(n)*m),THREADS,0,stream>>>(xq.data_ptr<float>(),cq.data_ptr<float>(),cross.data_ptr<float>(),n,m,r,low,storage);
  }else{
    auto stored=torch::empty({n,m},x.options().dtype(storage==0?type:torch::kFloat32));
    auto h=at::cuda::getCurrentCUDABlasHandle();detail::BlasState state(h,stream);
    EUCLIMIX_BLAS(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST));
    EUCLIMIX_BLAS(cublasSetMathMode(h,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    float alpha=1,beta=0;auto in=low==0?CUDA_R_16F:CUDA_R_16BF;auto out=storage==0?in:CUDA_R_32F;
    // Row-major N x M = transpose of column-major M x N. Quantized C comes first.
    EUCLIMIX_BLAS(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,r,&alpha,cl.data_ptr(),in,r,
      xl.data_ptr(),in,r,&beta,stored.data_ptr(),out,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cross=stored.to(torch::kFloat32);
  }
  auto d=torch::empty({n,m},x.options().dtype(output==32?torch::kFloat32:torch::kFloat64));
  auto mask=torch::empty({n,m},x.options().dtype(torch::kUInt8));
  auto b=torch::empty({n,m},x.options().dtype(torch::kFloat64));
#define LAUNCH(O) \
  select_distances<O><<<blocks(int64_t(n)*m),THREADS,0,stream>>>(cross.data_ptr<float>(),pn.data_ptr<float>(),cn.data_ptr<float>(),pb.data_ptr<double>(),cb.data_ptr<double>(),d.data_ptr<O>(),mask.data_ptr<uint8_t>(),b.data_ptr<double>(),n,m,r,low,storage,threshold,rho,epsilon); \
  if(correction==64)recompute_distances<O,64><<<blocks(int64_t(n)*m),THREADS,0,stream>>>(x.data_ptr<float>(),c.data_ptr<float>(),d.data_ptr<O>(),mask.data_ptr<uint8_t>(),n,m,r); \
  else recompute_distances<O,32><<<blocks(int64_t(n)*m),THREADS,0,stream>>>(x.data_ptr<float>(),c.data_ptr<float>(),d.data_ptr<O>(),mask.data_ptr<uint8_t>(),n,m,r);
  if(output==32){LAUNCH(float)}else{LAUNCH(double)}
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,mask,b};
}
Tensor direct_distance(Tensor x,Tensor c,int precision){
  detail::check_inputs(x,c);c10::cuda::CUDAGuard guard(x.device());TORCH_CHECK(precision==32||precision==64,"invalid precision");
  detail::record_inputs(x,c);
  int n=x.size(0),m=c.size(0),r=x.size(1);auto stream=c10::cuda::getCurrentCUDAStream();
  auto out=torch::empty({n,m},x.options().dtype(precision==64?torch::kFloat64:torch::kFloat32));
  if (n == 0 || m == 0) return out;
  if(precision==64)recompute_distances<double,64><<<blocks(int64_t(n)*m),THREADS,0,stream>>>(x.data_ptr<float>(),c.data_ptr<float>(),out.data_ptr<double>(),nullptr,n,m,r);
  else recompute_distances<float,32><<<blocks(int64_t(n)*m),THREADS,0,stream>>>(x.data_ptr<float>(),c.data_ptr<float>(),out.data_ptr<float>(),nullptr,n,m,r);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

}  // namespace euclimix
