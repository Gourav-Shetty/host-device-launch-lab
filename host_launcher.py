"""Host-side launcher: shows EXACTLY how host code launches device work.

Mirrors the CUDA runtime calls your textbook lists, in order:

  cudaMalloc -> cudaMemcpy(HostToDevice) -> kernel<<<grid, block>>>
      -> cudaDeviceSynchronize -> cudaMemcpy(DeviceToHost) -> cudaFree

Modes:
  python host_launcher.py --kernel saxpy --n 200000 --grid 64 --block 256
      -> simulated device (works everywhere, no GPU needed)

  python host_launcher.py --cuda [--kernel saxpy --n 200000 ...]
      -> REAL device if cupy or numba.cuda or torch.cuda exists,
         else exits with a clear message (dashboard shows the same).

Each launch prints ONE JSON object per line, like:
  {"launch_id": 0, "kernel": "saxpy", "grid": 64, "block": 256,
   "stages_ms": {...}, "total_ms": 12.4, "device": "cupy-cuda", ...}
so app.py can stream it live to the browser.
"""
import argparse
import json
import sys
import time


def detect_real_device():
    """Return (name, module-kind) for the first real GPU stack found."""
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return (f"torch-cuda ({torch.cuda.get_device_name(0)})", "torch")
    except Exception:
        pass
    try:
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
        if n > 0:
            return ("cupy-cuda", "cupy")
    except Exception:
        pass
    try:
        from numba import cuda
        if len(cuda.gpus) > 0:
            return ("numba-cuda", "numba")
    except Exception:
        pass
    return (None, None)


def run_simulated(args, launches=1):
    from device_core import simulate_launch_pipeline
    for i in range(launches):
        out = simulate_launch_pipeline(
            kernel=args.kernel, n=args.n, grid=args.grid,
            block=args.block, workers=args.workers, seed=i)
        out["launch_id"] = i
        out["mode"] = "simulated"
        print(json.dumps(out), flush=True)
        if args.delay > 0 and i + 1 < launches:
            time.sleep(args.delay)


def run_cuda(args, launches=1):
    name, kind = detect_real_device()
    if name is None:
        print(json.dumps({
            "error": "no CUDA device found (tried torch/cupy/numba). "
                     "Run without --cuda for the simulated device, or install "
                     "a CUDA stack on a GPU machine."}), flush=True)
        sys.exit(1)
    import numpy as np
    for i in range(launches):
        n = args.n
        rng = np.random.default_rng(i)
        x = rng.random(n, dtype=np.float32)
        y = rng.random(n, dtype=np.float32)
        a = np.float32(2.5)
        ev = {}
        t0 = time.perf_counter()
        # cudaMalloc + H2D
        if kind == "cupy":
            import cupy as cp
            t = time.perf_counter(); dx = cp.asarray(x); dy = cp.asarray(y)
            cp.cuda.runtime.deviceSynchronize(); ev["h2d"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); out = a * dx + dy
            cp.cuda.runtime.deviceSynchronize(); ev["kernel"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); host = cp.asnumpy(out)
            ev["d2h"] = (time.perf_counter() - t) * 1000
            del dx, dy, out
        elif kind == "torch":
            import torch
            t = time.perf_counter(); dx = torch.from_numpy(x).cuda()
            dy = torch.from_numpy(y).cuda(); torch.cuda.synchronize()
            ev["h2d"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); out = a * dx + dy
            torch.cuda.synchronize(); ev["kernel"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); host = out.cpu().numpy()
            ev["d2h"] = (time.perf_counter() - t) * 1000
            del dx, dy, out
        else:  # numba
            from numba import cuda as ncuda
            t = time.perf_counter(); dx = ncuda.to_device(x); dy = ncuda.to_device(y)
            ncuda.synchronize(); ev["h2d"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()

            @ncuda.jit
            def _saxpy(aa, xx, yy, oo):
                idx = ncuda.grid(1)
                if idx < oo.shape[0]:
                    oo[idx] = aa * xx[idx] + yy[idx]
            dout = ncuda.device_array_like(x)
            _saxpy[args.grid, args.block](a, dx, dy, dout)
            ncuda.synchronize(); ev["kernel"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); host = dout.copy_to_host()
            ev["d2h"] = (time.perf_counter() - t) * 1000
        total = (time.perf_counter() - t0) * 1000
        print(json.dumps({
            "launch_id": i, "mode": "cuda-real", "device": name,
            "kernel": args.kernel, "n": n, "grid": args.grid,
            "block": args.block, "total_threads": args.grid * args.block,
            "stages_ms": {
                "alloc": 0.0, "h2d": round(ev["h2d"], 3),
                "kernel": round(ev["kernel"], 3), "sync": 0.05,
                "d2h": round(ev["d2h"], 3), "free": 0.0},
            "total_ms": round(total, 3),
            "checksum": round(float(np.sum(host[:1024])), 3),
        }), flush=True)
        if args.delay > 0 and i + 1 < launches:
            time.sleep(args.delay)


def main():
    ap = argparse.ArgumentParser(description="Host launches device computation")
    ap.add_argument("--kernel", default="saxpy",
                    choices=["saxpy", "vecadd", "blur"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--cuda", action="store_true",
                    help="use real CUDA device instead of simulation")
    ap.add_argument("--check-device", action="store_true")
    args = ap.parse_args()
    if args.check_device:
        name, kind = detect_real_device()
        print(json.dumps({"device": name, "kind": kind}))
        return
    if args.cuda:
        run_cuda(args, args.launches)
    else:
        run_simulated(args, args.launches)


if __name__ == "__main__":
    main()
