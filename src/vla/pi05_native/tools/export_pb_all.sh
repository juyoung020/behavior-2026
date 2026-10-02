#!/bin/bash
# Export the four 2025 1st-place submission checkpoints (IliaLarchenko/behavior_submission) one after another.
# Peak RAM per export is printed (the WSL box is shared: run when memory is free).
for n in "$@"; do
  out=/mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt$n.pi05w
  [ -f "$out" ] && { echo "exists $out"; continue; }
  free -g | awk 'NR==2{print "free GB before:", $7}'
  /usr/bin/time -f "ckpt$n peak RSS %M KB, %e s" nice -n 10 bash /mnt/c/behavior-2026/src/vla/pi05_native/tools/wsl_py.sh \
    /mnt/c/behavior-2026/src/vla/pi05_native/tools/export_weights.py --arch pi_behavior \
    --ckpt /home/juyoung/checkpoints/behavior_submission/checkpoint_$n --asset IliaLarchenko/behavior_224_rgb --out $out 2>&1 \
    | grep -v -i warn
done
