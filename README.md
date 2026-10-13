# HostLaunch Lab — how host code launches device computation

**Realtime application:** watch a host (CPU) program allocate, copy, launch, wait, and copy back — the full `kernel<<<grid, block>>>` story — as moving graphs.

**Same shape as your sensor-fusion project**, so it feels familiar:
`device_core.py` (= sensor sim) · `host_launcher.py` (= MPI driver) · `app.py` + SSE (= Flask live view) · benchmark (= speedup proof).

## 1. Quickstart — native Windows, no GPU needed (port 5001)

```powershell
cd host-device-launch-lab
python -m pip install flask numpy
python app.py
# open http://127.0.0.1:5001
#  - "Launch stream" -> live H2D -> kernel -> D2H pipeline + charts
#  - "Race: CPU vs device" -> pure-Python host loop vs device launch speedup
#  - "Try real GPU" -> probes torch/cupy/numba CUDA, else explains fallback
```

Headless proof (no browser):

```powershell
python host_launcher.py --n 50000 --launches 5 --delay 0
python host_launcher.py --check-device
```

Benchmark proves the launch matters:

```powershell
python -c "from device_core import run_cpu_vs_device; b=run_cpu_vs_device(n=200000); print(b)"
# e.g. cpu ~45ms vs device total ~6ms -> ~7x (kernel-only even higher)
```

## 2. Real CUDA — GPU machine / WSL (graded path)

On your laptop there is no NVIDIA GPU, so the dashboard uses the
simulated parallel device (threads). The code path is identical —
`host_launcher.py --cuda` prints the same JSON with real
H2D/kernel/D2H timings once a CUDA stack exists:

```bash
bash wsl-setup.sh
# or manually: python3 host_launcher.py --cuda --n 200000 --launches 5
```

## 3. What to tell your professor (30 sec)

1. Host = CPU Python in `host_launcher.py`. Device = GPU (simulated here, real CUDA via `--cuda`).
2. Every launch does `cudaMalloc -> cudaMemcpy H2D -> kernel<<<grid,block>>> -> sync -> cudaMemcpy D2H -> cudaFree` (`device_core.py:simulate_launch_pipeline`).
3. `grid * block` = total threads; coverage = N / threads (elements per thread).
4. Dashboard shows the 6-stage pipeline bar + kernel-vs-total history + host log + CPU-vs-device race.
5. Small N is launch-overhead bound (copies dominate); big N shows the device win — that's the expected discussion point.

## 4. Files

- `device_core.py` — kernels (saxpy/vecadd/blur) + full launch pipeline with per-stage ms
- `host_launcher.py` — host CLI: simulated by default, `--cuda` for torch/cupy/numba, `--check-device` probe
- `app.py` — Flask + SSE `/stream`, `/api/benchmark`, `/api/device-status`, `/api/cuda-run` (port 5001)
- `templates/index.html` — Chart.js live view: pipeline bar, history, stages, host log
- `wsl-setup.sh`, `requirements.txt`, `README.md`
