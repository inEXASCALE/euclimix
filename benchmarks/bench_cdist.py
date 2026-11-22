"""Compare euclimix and PyTorch for all-pairs Euclidean distance on CUDA.

Run from a checkout with CUDA-enabled PyTorch.
    PYTHONPATH=src python benchmarks/bench_cdist.py
    PYTHONPATH=src python benchmarks/bench_cdist.py --shape 2048x256x128 --data near --torch-direct
"""

import argparse
from functools import partial
from statistics import median
from time import perf_counter

import torch


DEFAULT_SHAPES = (
    (16, 16, 64),
    (1024, 256, 64),
    (8192, 1024, 128),
    (4096, 4096, 128),
    (2048, 2048, 256),
)
DIRECT_MODE = "donot_use_mm_for_euclid_dist"


def parse_shape(value):
    try:
        n, m, d = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NxMxD, for example 8192x1024x128") from exc
    if min(n, m, d) < 1 or n * m >= 2**31 or d >= 2**23:
        raise argparse.ArgumentTypeError("use positive sizes with N*M < 2**31 and D < 2**23")
    return n, m, d


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def make_inputs(shape, generator, data):
    n, m, d = shape
    options = {"device": generator.device, "generator": generator, "dtype": torch.float32}
    x = torch.randn(n, d, **options)
    y = torch.randn(m, d, **options)
    if data == "near":
        center = torch.randn(1, d, **options)
        x.mul_(1e-3).add_(center)
        y.mul_(1e-3).add_(center)
    return x, y


def sampled_reference(x, y, count, generator):
    total = x.size(0) * y.size(0)
    if total <= count:
        pairs = torch.arange(total, device=x.device)
    else:
        pairs = torch.randint(total, (count,), device=x.device, generator=generator)
    rows = pairs // y.size(0)
    cols = pairs % y.size(0)

    # Direct FP64 subtraction on sampled pairs; bound the temporary gather size.
    chunk_size = max(1, 262_144 // x.size(1))
    reference = []
    for row_chunk, col_chunk in zip(rows.split(chunk_size), cols.split(chunk_size)):
        delta = x[row_chunk].double() - y[col_chunk].double()
        reference.append(torch.linalg.vector_norm(delta, dim=1))
    return rows, cols, torch.cat(reference)


def time_methods(methods, warmup, iterations, repeats):
    synchronize = torch.cuda.current_stream().synchronize
    for call in methods.values():
        for _ in range(warmup):
            call()
        synchronize()

    names = list(methods)
    timings = {name: [] for name in names}
    for repeat in range(repeats):
        # Rotate order so the same method does not always run first.
        start = repeat % len(names)
        for name in names[start:] + names[:start]:
            call = methods[name]
            synchronize()
            t0 = perf_counter()
            for _ in range(iterations):
                call()
            synchronize()
            timings[name].append((perf_counter() - t0) * 1e3 / iterations)
    return timings


def run_case(shape, generator, args, euclimix):
    x, y = make_inputs(shape, generator, args.data)
    methods = {"cdist(default)": partial(torch.cdist, x, y, p=2)}
    if args.torch_direct:
        methods["cdist(direct)"] = partial(torch.cdist, x, y, p=2, compute_mode=DIRECT_MODE)
    if euclimix is not None:
        options = {"precision": args.precision, "correction": args.correction, "rho": args.rho}
        methods["euclimix"] = partial(euclimix.pairwise_distances, x, y, **options)

    n, m, d = shape
    print(f"\nN={n} M={m} D={d}", flush=True)
    timings = time_methods(methods, args.warmup, args.iterations, args.repeats)
    baseline = median(timings["cdist(default)"])

    fallback = {}
    if euclimix is not None:
        info = euclimix.pairwise_squared_distances(x, y, return_info=True, **options)
        fallback["euclimix"] = f"{info.fallback_fraction:.3%}"
        del info

    rows, cols, reference = sampled_reference(x, y, args.samples, generator)
    nonzero = reference > 0
    print(f"FP64 reference on {reference.numel():,} pairs")
    print(f"{'method':<16} {'median ms':>10} {'min ms':>10} {'max ms':>10} "
          f"{'speedup':>9} {'max abs err':>12} {'max rel err':>12} {'fallback':>9}")
    for name, call in methods.items():
        samples = timings[name]
        elapsed = median(samples)
        error = (call()[rows, cols].double() - reference).abs()
        relative = error[nonzero] / reference[nonzero]
        max_relative = f"{relative.max().item():.3e}" if relative.numel() else "n/a"
        print(f"{name:<16} {elapsed:10.4f} {min(samples):10.4f} {max(samples):10.4f} "
              f"{baseline / elapsed:8.2f}x {error.max().item():12.3e} "
              f"{max_relative:>12} {fallback.get(name, '-'):>9}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_shapes = ", ".join("x".join(map(str, shape)) for shape in DEFAULT_SHAPES)
    parser.add_argument("--shape", type=parse_shape, action="append", metavar="NxMxD",
                        help=f"repeat for multiple cases (defaults are {default_shapes})")
    parser.add_argument("--data", choices=("random", "near"), default="random",
                        help="independent normal inputs or inputs near a common center")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--correction", choices=("fp32", "fp64", "same"), default="fp32",
                        help="precision for recomputing rejected pairs (default is %(default)s)")
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--matmul-precision", choices=("ieee", "tf32"), default="ieee",
                        help="PyTorch CUDA FP32 matmul policy (default: %(default)s)")
    parser.add_argument("--torch-direct", action="store_true",
                        help="also time cdist with matrix multiplication disabled")
    parser.add_argument("--torch-only", action="store_true", help="run PyTorch baselines only")
    parser.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=10, help="calls per timed repeat")
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--samples", type=positive_int, default=4096, help="maximum FP64 reference pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 1 < args.rho <= 1e6:
        parser.error("--rho must be in (1, 1e6]")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run with CUDA-enabled PyTorch and an NVIDIA GPU.")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device is outside the visible CUDA device range")

    device = torch.device("cuda", args.device)
    gpu = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    euclimix = None
    if not args.torch_only:
        if (major, minor) < (7, 5):
            parser.error(f"{gpu} is SM{major}{minor}. euclimix requires SM75+. "
                         "Use --torch-only for the PyTorch baseline.")
        if args.precision == "bf16" and major < 8:
            parser.error("BF16 requires SM80+. Use --precision fp16 on this GPU.")
        try:
            import euclimix
        except ModuleNotFoundError as exc:
            if exc.name != "euclimix":
                raise
            parser.error("Install euclimix or run from the checkout with PYTHONPATH=src.")

    torch.backends.cuda.matmul.fp32_precision = args.matmul_precision
    print(f"{gpu} (SM{major}{minor}) | PyTorch {torch.__version__} | CUDA {torch.version.cuda} "
          f"| FP32 matmul={torch.backends.cuda.matmul.fp32_precision}")
    print(f"FP32 inputs | data={args.data} | seed={args.seed}")
    if euclimix is not None:
        print(f"euclimix {euclimix.__version__} | precision={args.precision} "
              f"| correction={args.correction} | rho={args.rho:g}")
        print("The first warmup may compile the euclimix CUDA extension.")
    print(f"{args.warmup} warmups, {args.repeats} repeats of {args.iterations} calls. "
          "Per-call wall time from synchronized batches. Speedup is relative to cdist(default).", flush=True)

    with torch.cuda.device(device), torch.inference_mode():
        generator = torch.Generator(device=device)
        for shape in args.shape or DEFAULT_SHAPES:
            generator.manual_seed(args.seed)
            run_case(shape, generator, args, euclimix)


if __name__ == "__main__":
    main()
