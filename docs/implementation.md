# Numerical implementation

The mixed-precision functions evaluate the expanded squared-distance formula,
test each result, and recompute rejected entries using the direct formula.

## Original test with fused recomputation

`pairwise_squared_distances` supports FP16, BF16 and FP32 default paths.
FP16/BF16 inputs are quantized from the original FP32 coordinates. Squared
norms accumulate in FP32 and are stored in the selected format. Cross
products use cuBLAS GemmEx with FP32 compute and low-format output. The FP32
path uses LibTorch `mm` and its matrix-multiplication policy.

Distance formation, clamping to zero and threshold comparison use FP32.
The Original test recomputes distances at or below `rho * c(r) * (p + q)`,
where `p` and `q` are stored squared norms and `r` is the feature count:

| Precision / correction | `c(r)` |
|---|---|
| FP16 | `(2*r + 4) * 2**-11` |
| BF16 | `(2*r + 4) * 2**-7` |
| FP32 / FP32 or same | `(2*r + 4) * 2**-23` |
| FP32 / FP64 | `2*r * 2**-23` |

These are empirical coefficients. The BF16 coefficient uses `2**-7`,
whereas BF16 unit roundoff is `2**-8`.

One CUDA thread evaluates the Original test for each pair and applies the
direct formula to that pair when the test rejects it. FP32 and FP64 correction
read the original FP32 coordinates. The `same` option uses quantized
coordinates and low-format arithmetic. Nonfinite expanded results or
thresholds also trigger recomputation. The squared-distance API can report the
rejection mask. The Euclidean API takes the round-to-nearest FP32 square root
of the FP32 squared result in the same kernel, including for subnormal values.

## Mask-based implementation

`controlled_squared_distances` supports FP16/BF16 inputs, low/FP32
intermediate storage and Original/Extended tests. Norms use sequential FP32
FMA. Cross products use either cuBLAS or coordinate-order FP32 FMA.

The Extended bound includes input-conversion residuals, inner-product
arithmetic, intermediate storage and gradual underflow. Bounds and thresholds
are evaluated conservatively using directed FP64 rounding. The test accepts
a finite squared distance `d` only when `d > upward_round(rho * B)`.
Under the inner-product error assumption, accepted distances have relative
error at most `1/(rho-1)`. For cuBLAS, the coefficient is an assumption about
the library reduction; FP32 compute alone does not establish it.

Both tests compute diagnostic bounds. Rejected entries are recomputed by
scanning a mask. FP32/FP64 correction reads original FP32 inputs, with FP32
output in the Python interface. The native interface also supports FP64 output.
The accepted-path bound does not cover rounding after direct recomputation.

## Execution

All native calls use a CUDA device guard and the current PyTorch stream.
Input storage is recorded on that stream. PyTorch configures the cuBLAS
handle's stream and workspace; euclimix preserves them and restores modified
math/pointer modes. Rebinding the stream would reset the workspace, as
documented by [cuBLAS](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublassetstream).

The Original-test extension uses `--use_fast_math`. The mask-based extension
uses `--ftz=false --fmad=false --prec-div=true --prec-sqrt=true`. Separate
modules keep these compiler settings independent. Temporary buffers and
outputs are managed by LibTorch tensors.

Algorithm and analysis: [arXiv:2407.12208](https://arxiv.org/abs/2407.12208).
