#!/bin/bash
# Real CUDA path for grading (WSL Ubuntu with NVIDIA GPU).
# On Windows you demo the simulated device; on a CUDA box the SAME
# host_launcher.py --cuda path lights up.
set -e
sudo apt update && sudo apt install -y python3-pip nvidia-cuda-toolkit
pip3 install flask numpy numba
echo "--- check ---"
nvidia-smi || true
python3 host_launcher.py --check-device
echo "graded proof:"
python3 host_launcher.py --cuda --n 200000 --launches 5 --delay 0
