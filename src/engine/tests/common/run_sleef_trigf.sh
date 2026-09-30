#!/usr/bin/env bash
# sleef_trigf.h vs torch 전수 비교 (CPU)
set -e
B=~/engine-build/common-sleef
mkdir -p $B
g++ -O2 -ffp-contract=off -fno-fast-math -std=c++17 -fPIC -shared -I/mnt/c/behavior-2026/src/engine \
  /mnt/c/behavior-2026/src/engine/tests/common/sleef_capi.cpp -o $B/libsleef_t.so
~/behavior-linux/.venv/bin/python /mnt/c/behavior-2026/src/engine/tests/common/test_sleef_trigf.py $B/libsleef_t.so
