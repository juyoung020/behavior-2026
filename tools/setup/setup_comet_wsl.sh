#!/usr/bin/env bash
# 2위 Comet(openpi-comet) 실행용 사본을 WSL ~/openpi-comet 에 받고 README 대로 uv 환경을 만든다.
# - 커밋 4bb2aa7 (refs\openpi-comet 과 같은 커밋), 로컬 브랜치 behavior-2026-run. push 하지 않는다.
# - README 의 "behavior for server deploy"(bddl·OmniGibson[eval] 설치)는 하지 않는다:
#   2025 OmniGibson 의 omnigibson.learning 을 부르는 3곳을 2026 기준으로 고쳐서 필요가 없다(실행기록_스펙_있는그대로.md).
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/setup/setup_comet_wsl.sh
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
LOG=/mnt/c/behavior-2026/logs/setup_comet_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) 시작"
if [ ! -d ~/openpi-comet ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mli0603/openpi-comet.git ~/openpi-comet
fi
cd ~/openpi-comet
git checkout -B behavior-2026-run 4bb2aa7
git log --oneline -1
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
.venv/bin/python -c "import jax, openai, json_repair, tenacity; print('jax', jax.__version__, 'openai', openai.__version__)"
echo "== $(date) 끝"
