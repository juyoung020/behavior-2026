#!/usr/bin/env bash
# WSL Ubuntu 22.04 준비: 사용자, 기본 도구, uv, openpi(π0.5 서버), Isaac-GR00T.
# root 로 실행: wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/behavior-2026/tools/setup/setup_wsl.sh
# ROS 2 Humble 단계는 쓰지 않아 뺐다.
set -euo pipefail
LOG=/mnt/c/behavior-2026/logs/wsl_setup_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
export DEBIAN_FRONTEND=noninteractive
U=juyoung
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "사용자 $U"
if ! id "$U" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "$U"
  echo "$U ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$U"   # 비밀번호는 나중에 `passwd` 로 정한다
  chmod 440 "/etc/sudoers.d/$U"
fi
printf '[user]\ndefault=%s\n' "$U" > /etc/wsl.conf

step "기본 도구"
apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release ca-certificates git git-lfs \
  build-essential cmake python3-pip unzip
add-apt-repository -y universe

step "uv"
sudo -u "$U" bash -lc 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

step "openpi (behavior 브랜치, π0.5 서버)"
sudo -u "$U" bash -lc '
  cd ~
  [ -d openpi ] || git clone --recurse-submodules -b behavior https://github.com/wensi-ai/openpi.git
  cd openpi
  export PATH=$HOME/.local/bin:$PATH
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
'

step "Isaac-GR00T (코드만)"
sudo -u "$U" bash -lc 'cd ~ && ([ -d Isaac-GR00T ] || git clone https://github.com/wensi-ai/Isaac-GR00T.git)'

step "끝"
df -h / | tail -1
echo "=== WSL SETUP DONE ==="
