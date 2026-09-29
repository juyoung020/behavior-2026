#!/bin/bash
# joints 층 0 == 층 1 CPU 시험 전부 빌드·실행 (WSL). 사용: bash run_cpu.sh [trials]
ulimit -c 0
set -e
N=${1:-3000}
bash /mnt/c/behavior-2026/src/engine/tests/joints/build.sh test_joints_prep test_joints_block4 test_joints_prep_1lane test_joints_block4_1lane test_tanf test_rigid_api test_scene_query
export LD_LIBRARY_PATH=~/engine-deps/physx-107.3-omni/physx/bin/linux.x86_64/checked
cd ~/engine-build/joints/joints
set +e
for t in test_joints_prep test_joints_prep_1lane; do echo "== $t"; ./$t --trials $N 2>&1 | tail -22; done
for t in test_joints_block4 test_joints_block4_1lane; do echo "== $t"; ./$t --trials $N 2>&1 | tail -14; done
for t in test_rigid_api test_scene_query; do echo "== $t"; ./$t 2>&1 | tail -16; done
if [ "$2" = "libm" ]; then echo "== test_tanf"; ./test_tanf 2>&1 | tail -8; fi
