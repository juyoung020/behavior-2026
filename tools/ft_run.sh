#!/bin/bash
# 학습 데이터 파이프라인 도구 실행기 (WSL). 사용: bash /mnt/c/behavior-2026/tools/ft_run.sh tools/ft_bench.py stages
#  - ~/openpi 의 uv 환경 파이썬으로, 우리 코드(src/vla/fasttrain)를 PYTHONPATH 에 얹어 돌린다.
#  - JAX 는 원래 학습처럼 GPU 를 쓰되 미리 잡지 않게 한다(워커도 openpi 가 같은 값을 넣는다, data_loader.py:534-539).
set -e
REPO=/mnt/c/behavior-2026
export PYTHONPATH="$REPO/src:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export HF_HUB_OFFLINE=1
export PATH="$HOME/openpi/.venv/bin:$PATH"   # ninja (C++ 확장 빌드)
export TF_CPP_MIN_LOG_LEVEL=2
export FT_WORK="${FT_WORK:-$HOME/fasttrain_work}"
cd ~/openpi
script="$1"; shift
case "$script" in /*) ;; *) script="$REPO/$script" ;; esac
exec ~/openpi/.venv/bin/python "$script" "$@"
