#!/bin/bash
# WSL 통합 노드 simlink 빌드(ROS 없음). scenemap 은 같은 프로세스 C ABI 로 링크한다(build.rs):
#   SCENEMAP_LIB_DIR=<libscenemap 폴더> 가 있으면 진짜, 없으면 가짜 구현(src/sim/integ/scenemap_stub/sm_stub.cpp).
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/sim/integ/build_simlink.sh [test]
# 산출물: ~/cargo-target/simlink/release/simlink  (빌드 산출물은 ext4 에)
set -e
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/cargo-target/simlink
cd /mnt/c/behavior-2026/src/sim/integ/simlink
if [ "$1" = "test" ]; then
  cargo test --release 2>&1 | tail -30
else
  cargo build --release 2>&1 | grep -E "^(warning|error)|-->|Finished" | head -60
  echo "simlink: $CARGO_TARGET_DIR/release/simlink (scenemap: ${SCENEMAP_LIB_DIR:-가짜 구현})"
fi
