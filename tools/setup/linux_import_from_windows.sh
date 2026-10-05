#!/usr/bin/env bash
# 리눅스에서 Windows 파티션·WSL 디스크의 데이터를 가져온다 — docs/Linux_설치.md 5절.
#   sudo bash linux_import_from_windows.sh mount      # Windows C: 를 읽기 전용으로 /mnt/win 에 붙인다
#   bash      linux_import_from_windows.sh datasets   # BEHAVIOR-1K/datasets(약 35 GB, 복호화 키 포함) 복사
#   bash      linux_import_from_windows.sh data       # 우리 data/ 에서 필요한 것(시연 meta) 복사
#   sudo bash linux_import_from_windows.sh wsl        # WSL 디스크(ext4.vhdx)를 읽기 전용으로 /mnt/wsl 에 붙인다(체크포인트)
#   bash      linux_import_from_windows.sh ckpts      # WSL 의 ~/checkpoints 복사
#   sudo bash linux_import_from_windows.sh umount
# Windows 쪽 준비: 빠른 시작(Fast Startup) 끄기, 최대 절전 끄기(powercfg /h off) — 안 그러면 NTFS 가 '더러운' 상태라 붙지 않거나 읽기 전용이 된다.
# 모두 읽기 전용으로만 붙인다(Windows 쪽 파일을 절대 안 바꾼다).
set -euo pipefail
STEP=${1:-}
WORK=${WORK:-$HOME/behavior-2026}
WIN=/mnt/win
WSLM=/mnt/wsl
WIN_USER_DIR="$WIN/Users/user one"

case "$STEP" in
mount)
  mkdir -p "$WIN"
  DEV=${WIN_DEV:-$(lsblk -rpo NAME,FSTYPE,SIZE | awk '$2=="ntfs"{print $1, $3}' | sort -k2 -h | tail -1 | cut -d' ' -f1)}
  echo "Windows 파티션(가장 큰 NTFS): $DEV  -- 다르면 WIN_DEV=/dev/nvme0n1p3 처럼 지정"
  mount -t ntfs3 -o ro "$DEV" "$WIN" 2>/dev/null || mount -t ntfs-3g -o ro "$DEV" "$WIN"
  ls "$WIN/behavior-2026" | head
  ;;
datasets)
  mkdir -p "$WORK/BEHAVIOR-1K/datasets"
  rsync -a --info=progress2 "$WIN/behavior-2026/BEHAVIOR-1K/datasets/" "$WORK/BEHAVIOR-1K/datasets/"
  ls -la "$WORK/BEHAVIOR-1K/datasets"
  ;;
data)
  mkdir -p "$WORK/data"
  for d in 2026-challenge-demos/meta; do
    [ -e "$WIN/behavior-2026/data/$d" ] && rsync -a --info=progress2 "$WIN/behavior-2026/data/$d" "$WORK/data/$(dirname "$d")/"
  done
  # 개발 기록(재생 행동열·비교 기준)
  mkdir -p "$WORK/outputs"
  rsync -a "$WIN/behavior-2026/outputs/eval_turning_on_radio_20260929_195500_nf_a" "$WORK/outputs/"
  ;;
wsl)
  VHDX=${WSL_VHDX:-$(ls "$WIN_USER_DIR"/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu22.04*/LocalState/ext4.vhdx 2>/dev/null | head -1)}
  [ -n "$VHDX" ] || { echo "ext4.vhdx 를 못 찾음 — WSL_VHDX=경로 로 지정 (Windows: wsl -l -v, 레지스트리 Lxss BasePath)"; exit 1; }
  modprobe nbd max_part=8
  qemu-nbd -r -c /dev/nbd0 "$VHDX"
  sleep 1
  mkdir -p "$WSLM"
  mount -o ro,noload /dev/nbd0 "$WSLM" 2>/dev/null || mount -o ro,noload /dev/nbd0p1 "$WSLM"
  ls "$WSLM/home"
  ;;
ckpts)
  mkdir -p "$HOME/checkpoints"
  rsync -a --info=progress2 "$WSLM/home/juyoung/checkpoints/" "$HOME/checkpoints/"
  ;;
umount)
  umount "$WSLM" 2>/dev/null || true; qemu-nbd -d /dev/nbd0 2>/dev/null || true
  umount "$WIN" 2>/dev/null || true
  ;;
*)
  sed -n 2,11p "$0"; exit 2 ;;
esac
