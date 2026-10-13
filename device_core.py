"""Simulated device computation + host-launch pipeline timing.

Teaching model (mirrors CUDA runtime):
  host: cudaMalloc -> cudaMemcpy H2D -> kernel<<<grid,block>>> ->
        cudaDeviceSynchronize -> cudaMemcpy D2H -> cudaFree

Runs TODAY on native Windows with numpy + ThreadPoolExecutor as the
"device" (many SMs in parallel). If cupy / numba.cuda / torch.cuda
exists, host_launcher.py uses it as the REAL device; this file stays
the portable fallback so the classroom demo never dies.

Kernels offered:
  - saxpy  : Y = a*X + Y   (classic first CUDA sample)
  - vecadd : C = A + B
  - blur   : 3-point stencil (stand-in for image blur tile)
"""
import math
import time

import numpy as np


KERNELS = ("saxpy", "vecadd", "blur")


def make_inputs(kernel: str, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    if kernel == "saxpy":
        x = rng.random(n, dtype=np.float32)
        y = rng.random(n, dtype=np.float32)
        return {"x": x, "y": y, "a": np.float32(2.5)}
    if kernel == "vecadd":
        a = rng.random(n, dtype=np.float32)
        b = rng.random(n, dtype=np.float32)
        return {"a": a, "b": b}
    # blur: 1-D signal + 3-point stencil
    sig = rng.random(n, dtype=np.float32) * 100
    return {"sig": sig}


def cpu_reference(kernel: str, inputs: dict):
    """Slow pure-Python loop = 'host-only, no device' baseline."""
    t0 = time.perf_counter()
    if kernel == "saxpy":
        x, y, a = inputs["x"], inputs["y"], float(inputs["a"])
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = a * float(x[i]) + float(y[i])
    elif kernel == "vecadd":
        a, b = inputs["a"], inputs["b"]
        out = np.empty_like(a)
        for i in range(len(a)):
            out[i] = float(a[i]) + float(b[i])
    else:
        s = inputs["sig"]
        out = np.empty_like(s)
        n = len(s)
        for i in range(n):
            l = float(s[i - 1]) if i > 0 else float(s[0])
            c = float(s[i])
            r = float(s[i + 1]) if i + 1 < n else float(s[-1])
            out[i] = (l + 2 * c + r) / 4.0
    dt = (time.perf_counter() - t0) * 1000
    return out, dt


def device_kernel_numpy(kernel: str, inputs: dict):
    """Fast numpy path = 'device did it in parallel' stand-in."""
    t0 = time.perf_counter()
    if kernel == "saxpy":
        out = inputs["a"] * inputs["x"] + inputs["y"]
    elif kernel == "vecadd":
        out = inputs["a"] + inputs["b"]
    else:
        s = inputs["sig"]
        out = (np.roll(s, 1) + 2 * s + np.roll(s, -1)) / 4.0
        out[0] = (s[0] * 3 + s[1]) / 4.0
        out[-1] = (s[-2] + s[-1] * 3) / 4.0
    # copy to force eval timing
    out = np.array(out, dtype=np.float32, copy=True)
    dt = (time.perf_counter() - t0) * 1000
    return out, dt


def device_kernel_threaded(kernel: str, inputs: dict, workers: int = 8):
    """Thread-pool path = visible 'many SMs' parallel device."""
    from concurrent.futures import ThreadPoolExecutor

    n = next(v.shape[0] for v in inputs.values()
               if isinstance(v, np.ndarray))
    chunk = math.ceil(n / max(1, workers))
    t0 = time.perf_counter()
    scalar_a = None
    for v in inputs.values():
        if not isinstance(v, np.ndarray):
            try:
                scalar_a = float(v)
            except Exception:
                pass

    def work(lo: int, hi: int):
        if kernel == "saxpy":
            aa = scalar_a if scalar_a is not None else float(inputs.get("a", 2.5))
            return aa * inputs["x"][lo:hi] + inputs["y"][lo:hi]
        if kernel == "vecadd":
            return inputs["a"][lo:hi] + inputs["b"][lo:hi]
        s = inputs["sig"]
        seg = s[max(0, lo - 1): hi + 1]
        o = (np.roll(seg, 1) + 2 * seg + np.roll(seg, -1)) / 4.0
        # slice back to [lo, hi)
        start = lo - max(0, lo - 1)
        return o[start: start + (hi - lo)].astype(np.float32)

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        for lo in range(0, n, chunk):
            futs.append(pool.submit(work, lo, min(n, lo + chunk)))
        for f in futs:
            parts.append(f.result())
    out = np.concatenate(parts).astype(np.float32)
    dt = (time.perf_counter() - t0) * 1000
    return out, dt


def simulate_launch_pipeline(kernel="saxpy", n=200000, grid=64, block=256,
                             workers=8, seed=0):
    """Full host->device launch with per-stage timings (all simulated).

    Returns dict with stage_ms breakdown + result checksum, shaped like
    what host_launcher.py prints for a REAL CUDA run so the dashboard
    can render both identically.
    """
    total_threads = grid * block
    inputs = make_inputs(kernel, n, seed)
    elem_bytes = 4  # float32
    nbytes = n * elem_bytes * (2 if kernel != "saxpy" else 2)

    # 1. cudaMalloc (simulated: allocate device buffers)
    t0 = time.perf_counter()
    dev_buf = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            dev_buf[k] = np.empty_like(v)
        else:
            dev_buf[k] = v  # scalars (e.g. alpha) need no alloc, just copy
    alloc_ms = (time.perf_counter() - t0) * 1000

    # 2. cudaMemcpy H2D (simulated: host->device copy)
    t0 = time.perf_counter()
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            np.copyto(dev_buf[k], v)
    h2d_ms = (time.perf_counter() - t0) * 1000
    # model PCIe-ish floor so tiny N still shows a visible bar
    h2d_ms = max(h2d_ms, nbytes / 1e9 * 1000 * 0.05)

    # 3. kernel<<<grid, block>>> on device
    out, kernel_ms = device_kernel_threaded(kernel, dev_buf, workers=workers)

    # 4. cudaDeviceSynchronize (thread join already done; tiny fixed cost)
    sync_ms = 0.05

    # 5. cudaMemcpy D2H
    t0 = time.perf_counter()
    host_out = np.array(out, copy=True)
    d2h_ms = (time.perf_counter() - t0) * 1000
    d2h_ms = max(d2h_ms, n * elem_bytes / 1e9 * 1000 * 0.05)

    # 6. cudaFree (release device bufs)
    t0 = time.perf_counter()
    del dev_buf
    free_ms = (time.perf_counter() - t0) * 1000

    total_ms = alloc_ms + h2d_ms + kernel_ms + sync_ms + d2h_ms + free_ms
    checksum = round(float(np.sum(host_out[:min(n, 1024)])), 3)
    gflops = None
    flops = n * (2 if kernel in ("saxpy", "vecadd") else 4)
    if kernel_ms > 0:
        gflops = round(flops / (kernel_ms / 1000) / 1e9, 3)

    return {
        "kernel": kernel,
        "n": n,
        "grid": grid,
        "block": block,
        "total_threads": total_threads,
        "coverage": round(n / total_threads, 2) if total_threads else 0,
        "workers": workers,
        "stages_ms": {
            "alloc": round(alloc_ms, 3),
            "h2d": round(h2d_ms, 3),
            "kernel": round(kernel_ms, 3),
            "sync": round(sync_ms, 3),
            "d2h": round(d2h_ms, 3),
            "free": round(free_ms, 3),
        },
        "total_ms": round(total_ms, 3),
        "checksum": checksum,
        "gflops_device": gflops,
        "device": "simulated-parallel-device (threads)",
    }


def run_cpu_vs_device(kernel="saxpy", n=200000, grid=64, block=256,
                      workers=8, seed=0):
    """Benchmark: pure-Python host loop vs simulated device launch."""
    inputs = make_inputs(kernel, n, seed)
    _, cpu_ms = cpu_reference(kernel, inputs)
    launch = simulate_launch_pipeline(kernel, n, grid, block, workers, seed)
    speedup = round(cpu_ms / launch["total_ms"], 2) if launch["total_ms"] else 0
    k_speedup = round(cpu_ms / launch["stages_ms"]["kernel"], 2) \
        if launch["stages_ms"]["kernel"] else 0
    return {
        "cpu_ms": round(cpu_ms, 3),
        "device_total_ms": launch["total_ms"],
        "device_kernel_ms": launch["stages_ms"]["kernel"],
        "speedup_total": speedup,
        "speedup_kernel_only": k_speedup,
        "launch": launch,
    }
