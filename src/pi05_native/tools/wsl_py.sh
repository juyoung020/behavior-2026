#!/bin/bash
# Run a Python tool inside the WSL openpi venv (offline tools only: export / JAX reference dumps).
# usage: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/pi05_native/tools/wsl_py.sh <script.py> [args...]
# JAX_PLATFORMS defaults to cpu so nothing touches the GPU unless the caller sets JAX_PLATFORMS=cuda.
set -e
cd ~/openpi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python "$@"
