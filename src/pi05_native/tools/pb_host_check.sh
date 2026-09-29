#!/bin/bash
# CPU only: PiBehavior host side (state, tokens, transforms, cubic, correction rules, voting) vs the 1st-place Python code
set -e
D=/mnt/c/behavior-2026/src/pi05_native; B=~/pi05_native_build; mkdir -p $B
C=/mnt/c/behavior-2026/data/pi05_native/pb_host_cases.pi05d
bash $D/tools/wsl_py.sh $D/tools/make_pb_host_cases.py --out $C 2>&1 | grep -v -i warn
g++ -O2 -std=c++20 -o $B/pb_host_test $D/tools/pb_host_test.cpp $D/src/pb_host.cpp $D/src/weights.cpp
$B/pb_host_test /mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w $C
