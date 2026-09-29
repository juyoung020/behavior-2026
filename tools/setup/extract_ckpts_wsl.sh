#!/usr/bin/env bash
# 기본 제공 체크포인트 zip 을 WSL ~/checkpoints 에 풀고, 파일 수가 맞으면 Windows 쪽 zip 을 지운다.
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/setup/extract_ckpts_wsl.sh
set -euo pipefail
mkdir -p ~/checkpoints
cd ~/checkpoints
extract() {
  local name=$1 zip=$2
  echo "== $name"
  [ -f "$zip" ] || { echo "zip 없음: $zip"; return 0; }
  unzip -q -o "$zip" -d "$name"
  local want got
  want=$(unzip -Z1 "$zip" | grep -v '/$' | wc -l)
  got=$(find "$name" -type f | wc -l)
  echo "files zip=$want extracted=$got"
  du -sh "$name"
  if [ "$want" = "$got" ]; then rm -f "$zip"; echo "zip removed: $zip"; fi
}
extract pi05_turning_on_radio /mnt/c/behavior-2026/checkpoints/pi05_turning_on_radio/pi05TurningOnRadio.zip
extract groot_n17_turning_on_radio /mnt/c/behavior-2026/checkpoints/groot_n17_turning_on_radio/turning_on_radio_GR00T-checkpoint-150000.zip
df -h / | tail -1
echo "=== extract done ==="
