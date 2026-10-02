#!/usr/bin/env bash
# GPU 잠금 (전원 공통 규칙, docs/엔진_자체구현.md 12.6). C:\behavior-2026\.gpu_lock 디렉터리를 만드는 것이 잠금이다.
# owner.txt (한 줄에 키=값): owner= purpose= start_epoch= end_epoch=(유닉스 초) vram_gb=   (+ 참고용 pid=)
# 오래됨 판정은 now_epoch > end_epoch + 900 하나뿐. 한 번에 30 분 이내, 끝나면 바로 푼다. 남의 잠금은 풀지 않는다.
# 선착순 대기열(09-29, tools\gpu_lock.ps1 과 같은 형식): 기다리는 쪽은 .gpu_queue/<대기 시작 epoch_ns 20자리 0 채움>_<owner> 표 파일을 두고,
#   앞 숫자를 정수로 비교해 내 표가 가장 앞일 때만 잠금을 시도한다.
#   표 내용 = owner= purpose= host=wsl pid= start_epoch= end_epoch= minutes= vram_gb=
#   기다리는 동안 end_epoch 를 now+60 으로 다시 적는다(GPU_LOCK_POLL 초마다, 45초 이하). end_epoch 가 지난 표(대기자가 죽음)는 누구나 지운다.
#   이미 줄 선 자리를 이어받으려면 GPU_QUEUE_TS=<20자리 숫자> 를 주고 부른다.
#   source /mnt/c/behavior-2026/src/sim/engine/scripts/gpu_lock.sh
#   gpu_lock_acquire "engine-lead" "Linux 공식 기록" 20 4     # owner purpose 분(최대 30) vram_gb
#   ... GPU 작업 ...
#   gpu_lock_release "engine-lead"
# Windows 판: tools\gpu_lock.ps1 (같은 형식·동작)
GPU_LOCK_DIR=/mnt/c/behavior-2026/.gpu_lock
GPU_QUEUE_DIR=/mnt/c/behavior-2026/.gpu_queue
GPU_LOCK_POLL=${GPU_LOCK_POLL:-20}

_gpu_lock_get() {  # 키 값 읽기
  grep -m1 "^$1=" "$GPU_LOCK_DIR/owner.txt" 2>/dev/null | cut -d= -f2- | tr -d '\r'
}

_gpu_queue_prune() {  # end_epoch 가 지난 표 지우기 (내 표는 방금 갱신했으므로 안 걸린다)
  # end_epoch 가 없는 옛 형식 표(19자리, 옛 gpu_lock.sh)는 파일 수정 시각으로: 300 초 넘게 갱신 없으면 지운다(옛 도구도 기다리는 동안 touch 한다)
  local t now e age
  now=$(date +%s)
  for t in "$GPU_QUEUE_DIR"/*; do
    [ -f "$t" ] || continue
    e=$(grep -m1 "^end_epoch=" "$t" 2>/dev/null | cut -d= -f2- | tr -d '\r')
    if [[ "$e" =~ ^[0-9]+$ ]]; then
      if [ "$now" -gt "$e" ]; then
        echo "[gpu_lock] 오래된 대기표 지움(end_epoch 지남): $(basename "$t")"
        rm -f "$t"
      fi
    else
      age=$((now - $(stat -c %Y "$t" 2>/dev/null || echo "$now")))
      if [ "$age" -gt 300 ]; then
        echo "[gpu_lock] 옛 형식 대기표 지움(${age}초 갱신 없음): $(basename "$t")"
        rm -f "$t"
      fi
    fi
  done
}

_gpu_queue_write() {  # _gpu_queue_write owner purpose start_epoch minutes vram_gb 파일 — 표 내용 (ps1 과 같은 키)
  local now
  now=$(date +%s)
  {
    echo "owner=$1"
    echo "purpose=$2"
    echo "host=wsl"
    echo "pid=$$"
    echo "start_epoch=$3"
    echo "end_epoch=$((now + 60))"
    echo "minutes=$4"
    echo "vram_gb=$5"
  } > "$6"
}

_gpu_queue_sorted() {  # 표 이름을 앞 숫자(정수) 순으로. sort -n 은 앞 0 을 무시하고 정수로 비교한다
  ls "$GPU_QUEUE_DIR" 2>/dev/null | grep -E '^[0-9]+_' | sort -t_ -k1,1n
}

gpu_lock_acquire() {
  local who=$1 what=$2 mins=${3:-20} vram=${4:-0}
  [ "$mins" -gt 30 ] && mins=30
  mkdir -p "$GPU_QUEUE_DIR"
  local ts name ticket start
  ts=${GPU_QUEUE_TS:-$(printf '%020d' "$(date +%s%N)")}
  name="${ts}_$who"
  ticket="$GPU_QUEUE_DIR/$name"
  start=$(date +%s)
  while true; do
    _gpu_queue_write "$who" "$what" "$start" "$mins" "$vram" "$ticket"
    _gpu_queue_prune
    local first
    first=$(_gpu_queue_sorted | head -1)
    if [ "$first" = "$name" ]; then
      if mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
        local now end
        now=$(date +%s)
        end=$((now + mins * 60))
        printf 'owner=%s\npurpose=%s\nstart_epoch=%s\nend_epoch=%s\nvram_gb=%s\npid=%s\n' "$who" "$what" "$now" "$end" "$vram" "$$" \
          > "$GPU_LOCK_DIR/owner.txt"
        rm -f "$ticket"
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
    fi
    local pos
    pos=$(_gpu_queue_sorted | grep -n -x -F "$name" | cut -d: -f1)
    echo "[gpu_lock] 대기 ${pos:-?}번째 / 사용 중: $(_gpu_lock_get owner) / $(_gpu_lock_get purpose) / 끝 $(date -d "@$(_gpu_lock_get end_epoch)" '+%H:%M' 2>/dev/null) — ${GPU_LOCK_POLL}초 뒤 다시"
    sleep "$GPU_LOCK_POLL"
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

gpu_queue_leave() {  # 기다리다 그만둘 때: gpu_queue_leave <owner> — 내 표만 지운다
  rm -f "$GPU_QUEUE_DIR"/*_"$1"
}
