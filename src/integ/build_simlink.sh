#!/bin/bash
# WSL 통합 노드 simlink 빌드(r2r: ROS 2 Humble + meridian_msgs 를 source 한 채로 — 메시지 바인딩을 빌드 때 만든다).
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/integ/build_simlink.sh [test]
# 산출물: ~/cargo-target/simlink/release/simlink  (빌드 산출물은 ext4 에)
set -e
source /opt/ros/humble/setup.bash
source ~/meridian_ws/install/setup.bash        # meridian_msgs (GraphUpdateEventDev)
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/cargo-target/simlink
cd /mnt/c/behavior-2026/src/integ/simlink
if [ "$1" = "test" ]; then
  cargo test --release 2>&1 | tail -30
else
  cargo build --release 2>&1 | grep -E "^(warning|error)|-->|Finished" | head -60
  echo "simlink: $CARGO_TARGET_DIR/release/simlink"
fi
