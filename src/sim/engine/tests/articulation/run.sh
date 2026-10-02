#!/bin/bash
# articulation 시험 실행 (WSL): bash run.sh <프로그램> [인자...]
export LD_LIBRARY_PATH=~/engine-deps/physx-107.3-omni/physx/bin/linux.x86_64/checked
cd ~/engine-build/articulation/articulation
P=$1; shift
./$P "$@"
