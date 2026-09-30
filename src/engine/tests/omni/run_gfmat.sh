#!/usr/bin/env bash
# gfmat.h vs usdrt.Gf 비트 대조 (CPU, Kit 없음)
set -e
B=~/engine-build/omni-gfmat
mkdir -p $B
g++ -O2 -ffp-contract=off -fno-fast-math -std=c++17 -fPIC -shared -I/mnt/c/behavior-2026/src/engine \
  /mnt/c/behavior-2026/src/engine/tests/omni/gfmat_capi.cpp -o $B/libgfmat.so
E=~/behavior-linux/.venv/lib/python3.11/site-packages/isaacsim/extscache
U=$(ls -d $E/omni.usd.libs-*); D=$(ls -d $E/usdrt.scenegraph-*)
PYL=$(dirname "$(find ~/.local/share/uv/python -name 'libpython3.11.so.1.0' | head -1)")
LD_LIBRARY_PATH=$PYL:$U/bin:$D/bin:${LD_LIBRARY_PATH:-} ~/behavior-linux/.venv/bin/python /mnt/c/behavior-2026/src/engine/tests/omni/test_gfmat.py $B/libgfmat.so
