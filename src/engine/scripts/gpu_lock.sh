#!/usr/bin/env bash
# GPU 잠금 (전원 공통 규칙, docs/엔진_자체구현.md 12.6). C:\behavior-2026\.gpu_lock 디렉터리를 만드는 것이 잠금이다.
# owner.txt (한 줄에 키=값): owner= purpose= start_epoch= end_epoch=(유닉스 초) vram_gb=   (+ 참고용 pid=)
# 오래됨 판정은 now_epoch > end_epoch + 900 하나뿐. 한 번에 30 분 이내, 끝나면 바로 푼다. 남의 잠금은 풀지 않는다.
#   source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
#   gpu_lock_acquire "engine-lead" "Linux 공식 기록" 20 4     # owner purpose 분(최대 30) vram_gb
#   ... GPU 작업 ...
#   gpu_lock_release "engine-lead"
# Windows 판: tools\gpu_lock.ps1 (같은 형식·동작)
GPU_LOCK_DIR=/mnt/c/behavior-2026/.gpu_lock

_gpu_lock_get() {  # 키 값 읽기
  grep -m1 "^$1=" "$GPU_LOCK_DIR/owner.txt" 2>/dev/null | cut -d= -f2- | tr -d '\r'
}

gpu_lock_acquire() {
  local who=$1 what=$2 mins=${3:-20} vram=${4:-0}
  [ "$mins" -gt 30 ] && mins=30
  while true; do
    if mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
      local now end
      now=$(date +%s)
      end=$((now + mins * 60))
      printf 'owner=%s\npurpose=%s\nstart_epoch=%s\nend_epoch=%s\nvram_gb=%s\npid=%s\n' "$who" "$what" "$now" "$end" "$vram" "$$" \
        > "$GPU_LOCK_DIR/owner.txt"
      echo "[gpu_lock] 잡음: $who ($what) ~ $(date -d "@$end" '+%H:%M:%S')"
      return 0
    fi
    local endep
    endep=$(_gpu_lock_get end_epoch)
    if [[ "$endep" =~ ^[0-9]+$ ]] && [ "$(date +%s)" -gt $((endep + 900)) ]; then
      echo "[gpu_lock] 오래된 잠금 지움: $(tr '\n' ' ' < "$GPU_LOCK_DIR/owner.txt")"
      rm -rf "$GPU_LOCK_DIR"
      continue
    fi
    echo "[gpu_lock] 사용 중: $(_gpu_lock_get owner) / $(_gpu_lock_get purpose) / 끝 $(date -d "@${endep:-0}" '+%H:%M' 2>/dev/null) — 45초 뒤 다시"
    sleep 45
  done
}

gpu_lock_release() {  # gpu_lock_release <owner> : 내 잠금일 때만 푼다
  local who=$1 cur
  cur=$(_gpu_lock_get owner)
  if [ -n "$cur" ] && [ -n "$who" ] && [ "$cur" != "$who" ]; then
    echo "[gpu_lock] 남의 잠금이라 안 푼다: $cur"
    return 1
  fi
  rm -rf "$GPU_LOCK_DIR" && echo "[gpu_lock] 풀음"
}
