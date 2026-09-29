#!/usr/bin/env bash
# 대회 환경(Linux) 공식 평가기를 이 PC 의 WSL2 Ubuntu 22.04 에 물리 전용으로 설치한다 (층 0 기준 뜨기용).
#   bash /mnt/c/behavior-2026/src/engine/scripts/setup_linux_official.sh [download|deps|isaac|all]
# 대회 원문: 성능 측정 장비 Ubuntu 22.04.5 + RTX 4090, 최종 평가 GPU 3090·A5000·TitanRTX.
# 이 스크립트는 BEHAVIOR-1K setup.sh 의 Linux 경로(pip isaacsim 5.1.0 휠 25개 + OmniGibson[eval] + bddl3)를 그대로 따른다.
#   - conda 대신 uv 로 파이썬 3.11 가상환경 (~/behavior-linux/.venv)
#   - BEHAVIOR-1K 는 Windows 쪽 원본을 건드리지 않게 같은 커밋(v3.9.3-post1, bd049de)을 WSL 안에 복제
#   - datasets 는 Windows 쪽을 그대로 읽는다: OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets (34.7 GB 복사 안 함)
set -euo pipefail
STEP=${1:-all}
BASE=~/behavior-linux
WHEELS=$BASE/wheels
SRC=/mnt/c/behavior-2026/BEHAVIOR-1K
COMMIT=bd049de3119acdcdf2334fe9e1ebe060fa20c108
mkdir -p "$BASE" "$WHEELS"
export OMNI_KIT_ACCEPT_EULA=YES

PKGS=(isaacsim_kernel-5.1.0.0 isaacsim_app-5.1.0.0 isaacsim_core-5.1.0.0 isaacsim_gui-5.1.0.0 isaacsim_utils-5.1.0.0
      isaacsim_storage-5.1.0.0 isaacsim_asset-5.1.0.0 isaacsim_sensor-5.1.0.0 isaacsim_robot_motion-5.1.0.0 isaacsim_robot-5.1.0.0
      isaacsim_benchmark-5.1.0.0 isaacsim_code_editor-5.1.0.0 isaacsim_ros1-5.1.0.0 isaacsim_cortex-5.1.0.0 isaacsim_example-5.1.0.0
      isaacsim_replicator-5.1.0.0 isaacsim_rl-5.1.0.0 isaacsim_robot_setup-5.1.0.0 isaacsim_ros2-5.1.0.0 isaacsim_template-5.1.0.0
      isaacsim_test-5.1.0.0 isaacsim-5.1.0.0 isaacsim_extscache_physics-5.1.0.0 isaacsim_extscache_kit-5.1.0.0
      isaacsim_extscache_kit_sdk-5.1.0.0)

download() {
  for pkg in "${PKGS[@]}"; do
    name=${pkg%-*}
    f="${pkg}-cp311-none-manylinux_2_35_x86_64.whl"
    url="https://pypi.nvidia.com/${name//_/-}/$f"
    want=$(curl -sSIL --fail "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub("\r","",v); print v}')
    if [ -f "$WHEELS/$f" ] && [ "$(stat -c %s "$WHEELS/$f")" = "$want" ]; then echo "있음 $f"; continue; fi
    echo "받는 중 $f ($((want/1000000)) MB)"
    curl -sSL --fail --retry 5 --speed-limit 1024 --speed-time 60 "$url" -o "$WHEELS/$f.part"
    mv "$WHEELS/$f.part" "$WHEELS/$f"
  done
}

deps() {  # isaacsim 휠 없이 되는 부분 (휠 받는 동안 먼저)
  cd "$BASE"
  if [ ! -d BEHAVIOR-1K/.git ]; then
    git clone --no-checkout "$SRC" BEHAVIOR-1K
    git -C BEHAVIOR-1K checkout -q "$COMMIT"
  fi
  [ -d .venv ] || ~/.local/bin/uv venv --python 3.11 .venv
  source .venv/bin/activate
  ~/.local/bin/uv pip install pip
  python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install "numpy<2"
  python -m pip install -e BEHAVIOR-1K/bddl3
  python -m pip install -e "BEHAVIOR-1K/OmniGibson[eval]" --no-build-isolation
}

isaac() {
  cd "$BASE"
  source .venv/bin/activate
  python -m pip install "$WHEELS"/*.whl
  ISAAC_PATH=$(python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))" 2>/dev/null | tail -1)
  if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/extscache" ]; then
    find "$ISAAC_PATH/extscache" -type d -name websockets -path "*/pip_prebundle/*" -exec rm -rf {} + 2>/dev/null || true
  fi
  python -m pip install --force-reinstall cffi==1.17.1
  python -m pip install --force-reinstall "websockets>=15.0.1"
  python -c "import isaacsim, omnigibson; print('import OK')"
}

case "$STEP" in
  download) download ;;
  deps) deps ;;
  isaac) isaac ;;
  all) download; deps; isaac ;;
esac
