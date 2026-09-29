#!/usr/bin/env bash
# Linux 대회 환경(WSL) 공통 환경 변수. source 해서 쓴다:  source /mnt/c/behavior-2026/src/engine/scripts/linux_env.sh
source ~/behavior-linux/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets
export OMNIGIBSON_APPDATA_PATH=~/behavior-linux/appdata
export PYTHONUTF8=1
ISAAC_PKG=$(python -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))' 2>/dev/null | tail -1)
export ISAAC_PKG
