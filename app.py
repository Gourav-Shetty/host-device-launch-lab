"""Flask dashboard: watch host code launch device computation live.

Native Windows (works TODAY, no GPU needed):
    pip install flask numpy
    python app.py  ->  http://127.0.0.1:5001

Real CUDA path (on a GPU machine / WSL with toolkit):
    python host_launcher.py --cuda --n 200000 --launches 5
    # or click "Try real GPU" in the dashboard (/api/cuda-run)

Endpoints:
    GET  /                        dashboard
    GET  /stream?...               SSE live launches (simulated device)
    POST /api/benchmark            {kernel,n,grid,block,workers} -> cpu vs device
    GET  /api/device-status        cpu + real-GPU stack presence
    POST /api/cuda-run             run host_launcher.py --cuda, return rows
"""
import json
import os
import subprocess
import sys
import time

from flask import Flask, Response, jsonify, render_template, request

from device_core import run_cpu_vs_device, simulate_launch_pipeline

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/device-status")
def device_status():
    import multiprocessing
    import numpy as np
    info = {
        "cpu_cores": multiprocessing.cpu_count(),
        "numpy": np.__version__,
        "torch_cuda": False,
        "cupy": False,
        "numba_cuda": False,
        "real_device": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda"] = True
            info["real_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import cupy  # noqa: F401
        info["cupy"] = True
    except Exception:
        pass
    try:
        from numba import cuda
        if len(cuda.gpus) > 0:
            info["numba_cuda"] = True
    except Exception:
        pass
    return jsonify(info)


@app.route("/stream")
def stream():
    kernel = request.args.get("kernel", "saxpy")
    if kernel not in ("saxpy", "vecadd", "blur"):
        kernel = "saxpy"
    n = max(1000, min(int(request.args.get("n", 200000)), 2000000))
    grid = max(1, min(int(request.args.get("grid", 64)), 512))
    block = max(32, min(int(request.args.get("block", 256)), 1024))
    launches = max(1, min(int(request.args.get("launches", 20)), 200))
    workers = max(1, min(int(request.args.get("workers", 8)), 16))

    def gen():
        for i in range(launches):
            out = simulate_launch_pipeline(kernel, n, grid, block,
                                           workers, seed=i)
            out["launch_id"] = i
            out["mode"] = "simulated"
            # plain-words explanation of THIS launch for the log
            s = out["stages_ms"]
            out["story"] = (
                f"Launch {i}: host mallocs {n} floats, copies H2D "
                f"({s['h2d']}ms), fires {grid}x{block}={grid*block} threads "
                f"for {kernel} ({s['kernel']}ms kernel), syncs, copies back "
                f"D2H ({s['d2h']}ms). Total {out['total_ms']}ms."
            )
            yield f"data: {json.dumps(out)}\n\n"
            time.sleep(0.35)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    body = request.get_json(force=True, silent=True) or {}
    kernel = body.get("kernel", "saxpy")
    if kernel not in ("saxpy", "vecadd", "blur"):
        kernel = "saxpy"
    n = max(1000, min(int(body.get("n", 200000)), 2000000))
    grid = max(1, min(int(body.get("grid", 64)), 512))
    block = max(32, min(int(body.get("block", 256)), 1024))
    workers = max(1, min(int(body.get("workers", 8)), 16))
    res = run_cpu_vs_device(kernel, n, grid, block, workers)
    res.update({"kernel": kernel, "n": n, "grid": grid,
                "block": block, "workers": workers,
                "note": "cpu_ms = pure-Python host loop. device_total_ms = "
                        "malloc+H2D+kernel+sync+D2H+free. Small N is "
                        "launch-overhead bound; big N shows the win."})
    return jsonify(res)


@app.route("/api/cuda-run", methods=["POST"])
def cuda_run():
    """Run the REAL device path and return its rows (or a clear error)."""
    body = request.get_json(force=True, silent=True) or {}
    kernel = body.get("kernel", "saxpy")
    n = max(1000, min(int(body.get("n", 200000)), 5000000))
    grid = max(1, min(int(body.get("grid", 64)), 1024))
    block = max(32, min(int(body.get("block", 256)), 1024))
    launches = max(1, min(int(body.get("launches", 5)), 20))
    cmd = [sys.executable, "host_launcher.py", "--cuda",
           "--kernel", kernel, "--n", str(n), "--grid", str(grid),
           "--block", str(block), "--launches", str(launches),
           "--delay", "0"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=180, cwd=BASE_DIR)
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "CUDA run timed out"}), 500
    rows = []
    for ln in (p.stdout or "").splitlines():
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except Exception:
                rows.append({"raw": ln})
    ok = p.returncode == 0 and not (rows and "error" in rows[0])
    return jsonify({"ok": ok, "returncode": p.returncode,
                    "stderr": (p.stderr or "")[-2000:], "rows": rows})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
