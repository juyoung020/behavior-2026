#!/usr/bin/env bash
# 리눅스 듀얼 부팅(Ubuntu 22.04) 첫 설치 — docs/Linux_설치.md 3~5절의 명령을 순서대로 묶은 것.
#   bash linux_setup.sh driver     # ① NVIDIA 드라이버(open 커널 모듈) — 끝나면 재부팅
#   bash linux_setup.sh base       # ② 기본 도구·miniconda·gh·Rust
#   bash linux_setup.sh repos      # ③ 우리 저장소 + BEHAVIOR-1K v3.9.3-post1 (bd049de)
#   bash linux_setup.sh behavior   # ④ 공식 setup.sh (conda env behavior, Isaac Sim 5.1 pip, OmniGibson[eval], bddl, joylo) — 데이터셋은 ⑤ 에서 옮김
#   bash linux_setup.sh check      # 설치 확인(드라이버·GPU·isaacsim import)
# 각 단계는 여러 번 돌려도 된다(이미 된 것은 건너뜀). sudo 비밀번호를 묻는다.
set -euo pipefail
STEP=${1:-}
DRIVER_BRANCH=${DRIVER_BRANCH:-580}        # RTX 5070 Ti(Blackwell)는 570 이상 + open 커널 모듈이 필요. Isaac Sim 5.1 은 580 계열 권장(추정 — 설치 뒤 nvidia-smi 로 확인)
WORK=${WORK:-$HOME/behavior-2026}
B1K_COMMIT=bd049de3119acdcdf2334fe9e1ebe060fa20c108   # v3.9.3-post1 (Windows 와 같은 커밋)

case "$STEP" in
driver)
  sudo apt-get update
  sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) ubuntu-drivers-common
  ubuntu-drivers devices || true
  # Blackwell 은 open 커널 모듈만 지원한다
  sudo apt-get install -y "nvidia-driver-${DRIVER_BRANCH}-open"
  echo "설치 끝. Secure Boot 가 켜져 있으면 재부팅 때 파란 화면(MOK)에서 'Enroll MOK' -> 설치 때 정한 비밀번호. 그 다음: sudo reboot"
  ;;
base)
  sudo apt-get install -y git git-lfs curl wget unzip ffmpeg htop nvtop qemu-utils ntfs-3g libvulkan1 vulkan-tools \
       libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6
  if ! command -v conda >/dev/null && [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$HOME/miniconda3" && rm /tmp/mc.sh
    "$HOME/miniconda3/bin/conda" init bash
  fi
  if ! command -v gh >/dev/null; then
    sudo mkdir -p -m 755 /etc/apt/keyrings
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update && sudo apt-get install -y gh
  fi
  command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  echo "다음: 새 셸을 열고 gh auth login (저장소가 비공개)"
  ;;
repos)
  mkdir -p "$(dirname "$WORK")"
  [ -d "$WORK/.git" ] || gh repo clone juyoung020/behavior-2026 "$WORK"
  git -C "$WORK" config user.name juyoung020
  git -C "$WORK" config user.email 151780134+juyoung020@users.noreply.github.com
  if [ ! -d "$WORK/BEHAVIOR-1K/.git" ]; then
    git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$WORK/BEHAVIOR-1K"
  fi
  git -C "$WORK/BEHAVIOR-1K" fetch --tags -q
  git -C "$WORK/BEHAVIOR-1K" checkout -q "$B1K_COMMIT"
  git -C "$WORK/BEHAVIOR-1K" describe --tags
  ;;
behavior)
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  cd "$WORK/BEHAVIOR-1K"
  # 공식 명령(평가 문서 docs/raw/site_challenge_evaluation.md: ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval).
  # 데이터셋(약 35 GB)은 받지 않고 Windows 쪽을 옮긴다(docs/Linux_설치.md 5절) -> --dataset 빼고 돌린다.
  ./setup.sh --new-env --omnigibson --bddl --joylo --eval \
      --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
  conda activate behavior
  python -m pip install av "numpy<2"   # tools/black_frame_check.py 가 쓴다
  ;;
check)
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" | head -4 || true
  source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate behavior
  python -c "import isaacsim, omnigibson, bddl; print('isaacsim', isaacsim.__file__); print('omnigibson', omnigibson.__file__)"
  ;;
*)
  sed -n 2,9p "$0"; exit 2 ;;
esac
