#!/usr/bin/env bash
# behavior 환경 보수: setup.ps1 이 로그 없이 실패한 단계들을 다시 한다.
#  - torch 가 CPU 판(2.7.0+cpu)으로 깔려 있었다 -> cu128 판으로 교체 (RTX 5070 Ti = sm_120 은 cu128 필요)
#  - bddl3, OmniGibson[eval] 이 설치돼 있지 않았다
# 실행 (Git Bash): bash /c/behavior-2026/tools/setup/fix_behavior_env.sh
set -euo pipefail
PY="/c/Users/user one/anaconda3/envs/behavior/python.exe"
ROOT=/c/behavior-2026/BEHAVIOR-1K
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMNI_KIT_ACCEPT_EULA=YES
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "torch 2.7.0 cu128"
"$PY" -m pip install --force-reinstall --no-deps torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

step "torch-cluster (torch 2.7.0+cu128 용)"
"$PY" -m pip install --force-reinstall --no-deps torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

step "bddl3"
"$PY" -m pip install -e "C:/behavior-2026/BEHAVIOR-1K/bddl3"

step "OmniGibson[eval]"
"$PY" -m pip install -e "C:/behavior-2026/BEHAVIOR-1K/OmniGibson[eval]"

step "torch 가 다시 바뀌지 않았나"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
"$PY" -m pip install --force-reinstall --no-deps cffi==1.17.1
"$PY" -c "import omnigibson, bddl; print('omnigibson', omnigibson.__file__); print('bddl', bddl.__file__)"
echo "=== FIX DONE ==="
