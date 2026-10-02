#!/usr/bin/env bash
# 공용 GPU 잠금 (전원 공통 규칙, 리눅스판; 옛 Windows 판 archive/tools/gpu_lock.ps1 과 같은 형식·동작).
# <저장소>/.gpu_lock 디렉터리를 만드는 것이 곧 잠금. 선착순 대기열 <저장소>/.gpu_queue/.
#   (엔진의 src/sim/engine/scripts/gpu_lock.sh 가 쓰는 /mnt/c/behavior-2026/.gpu_lock 는 이 PC 에서 같은 디렉터리를 가리킨다 — 심볼릭 링크)
# owner.txt 형식(전원 공통): owner= purpose= start_epoch= end_epoch=(유닉스 초) vram_gb=   (+ 참고용 pid=)
# 오래됨 판정은 epoch 초 계산만: now_epoch > end_epoch + 900. end_epoch 를 못 읽으면 지우지 않는다.
#
#   명령:  tools/gpu_lock.sh acquire --owner fasteval --purpose '평가 3 판' --minutes 10 --vram-gb 6 [--max-wait-min 120]
#          tools/gpu_lock.sh release --owner fasteval     (owner 가 같을 때만 푼다)
#          tools/gpu_lock.sh status                       (잠금 + 대기열)
#   함수:  source tools/gpu_lock.sh ; gpu_lock_enter <owner> <purpose> <minutes> <vram_gb> [max_wait_min] ; gpu_lock_exit <owner>
#          gpu_lock_cancel : 기다리다 끊을 때 내 표 지우기 (예: trap 'gpu_lock_cancel' INT TERM)
#          (source 해도 부르는 셸의 set 옵션은 안 바꾼다)
# 대기열(선착순)
# - 줄 서기: .gpu_queue/<start_epoch_ns 20자리>_<owner> 표 파일(내용 owner= purpose= host=linux pid= start_epoch= end_epoch= minutes= vram_gb=).
#   기다리는 동안 GPU_LOCK_POLL 초(기본 45)마다 표의 end_epoch 를 now + 60 으로 새로 적는다(살아 있다는 표시).
# - 내 표가 이름 앞 숫자 순으로 가장 앞일 때만 잠금 디렉터리를 만든다. 잡으면 내 표를 지운다. 못 잡고 포기해도 내 표를 지운다.
# - 오래된 표: now_epoch > end_epoch + 900, 또는 host=linux|wsl 인데 그 pid 프로세스가 없음 -> 지운다
#   (이 PC 에서는 엔진 gpu_lock.sh 의 host=wsl 표도 같은 기계의 pid 다).
# - 잠금 주인과 owner 가 같은 가장 앞 표는 "잡았음" 으로 보고 지운다(대기열을 모르는 도구로 잡은 경우).
# - 한 번에 30 분 이내(넘게 주면 30 으로 자름). 잡은 뒤에도 VRAM 에 민감한 일은 nvidia-smi 로 다른 프로세스가 없는지 따로 확인한다.

_GPU_LOCK_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$_GPU_LOCK_REPO/.gpu_lock}
GPU_QUEUE_DIR=${GPU_QUEUE_DIR:-$_GPU_LOCK_REPO/.gpu_queue}
GPU_LOCK_POLL=${GPU_LOCK_POLL:-45}

_gl_now() { date +%s; }

_gl_key() {  # _gl_key <파일> <키> : 키 값 (없으면 빈 문자열)
  [ -f "$1" ] || return 0
  grep -m1 -E "^[[:space:]]*$2[[:space:]]*=" "$1" 2>/dev/null | cut -d= -f2- | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true
}

_gl_epoch_stale() {  # _gl_epoch_stale <파일> : end_epoch + 900 초가 지났으면 0(참)
  local e
  e=$(_gl_key "$1" end_epoch)
  [[ "$e" =~ ^[0-9]+$ ]] || return 1
  [ "$(_gl_now)" -gt $((e + 900)) ]
}

_gl_tickets() {  # 표 경로를 이름 앞 숫자(정수) 순으로
  [ -d "$GPU_QUEUE_DIR" ] || return 0
  local f n
  for f in "$GPU_QUEUE_DIR"/*; do
    n=${f##*/}
    [ -f "$f" ] && [[ "$n" =~ ^[0-9]+_ ]] && [[ "$n" != *.tmp* ]] && echo "$n"
  done | sort -t_ -k1,1n -k2 | while read -r n; do echo "$GPU_QUEUE_DIR/$n"; done || true
}

_gl_write_ticket() {  # _gl_write_ticket <파일> owner purpose start minutes vram
  local tmp="$1.tmp$$"
  printf 'owner=%s\npurpose=%s\nhost=linux\npid=%s\nstart_epoch=%s\nend_epoch=%s\nminutes=%s\nvram_gb=%s\n' \
    "$2" "$3" "$$" "$4" "$(($(_gl_now) + 60))" "$5" "$6" > "$tmp" && mv -f "$tmp" "$1"
}

_gl_prune() {  # _gl_prune <내 표> : 오래된 표 지우기
  local mine=$1 holder t o h p dead
  holder=$(_gl_key "$GPU_LOCK_DIR/owner.txt" owner)
  while read -r t; do
    [ -n "$t" ] || continue
    [ "$t" = "$mine" ] && continue
    o=$(_gl_key "$t" owner); h=$(_gl_key "$t" host); p=$(_gl_key "$t" pid)
    dead=0
    if [ -n "$holder" ] && [ "$o" = "$holder" ]; then dead=1; holder=
    elif _gl_epoch_stale "$t"; then dead=1
    elif { [ "$h" = linux ] || [ "$h" = wsl ]; } && [[ "$p" =~ ^[0-9]+$ ]] && [ ! -d "/proc/$p" ]; then dead=1
    fi
    if [ $dead = 1 ]; then echo "[gpu_lock] 오래된 표 지움: $(basename "$t")"; rm -f "$t"; fi
  done < <(_gl_tickets)
}

gpu_lock_enter() {  # gpu_lock_enter <owner> <purpose> [minutes=30] [vram_gb=?] [max_wait_min=120] : 잡으면 0, 못 잡으면 1
  local owner=$1 purpose=${2:-} minutes=${3:-30} vram=${4:-?} maxwait=${5:-120}
  [ "$minutes" -gt 30 ] && minutes=30
  local t0 safe mine q pos s o left ahead held i
  t0=$(_gl_now)
  mkdir -p "$GPU_QUEUE_DIR"
  safe=$(printf '%s' "$owner" | sed 's/[^A-Za-z0-9._-]/-/g')
  mine="$GPU_QUEUE_DIR/$(printf '%020d' "$(date +%s%N)")_$safe"
  _GL_MINE=$mine
  while true; do
    _gl_write_ticket "$mine" "$owner" "$purpose" "$t0" "$minutes" "$vram"
    _gl_prune "$mine"
    mapfile -t q < <(_gl_tickets)
    pos=-1
    for i in "${!q[@]}"; do [ "${q[$i]}" = "$mine" ] && { pos=$i; break; }; done
    if [ "$pos" = 0 ]; then
      if mkdir "$GPU_LOCK_DIR" 2>/dev/null; then  # mkdir 한 번 -- 이미 있으면 실패(원자적)
        s=$(_gl_now)
        printf 'owner=%s\npurpose=%s\nstart_epoch=%s\nend_epoch=%s\nvram_gb=%s\npid=%s\n' \
          "$owner" "$purpose" "$s" "$((s + 60 * minutes))" "$vram" "$$" > "$GPU_LOCK_DIR/owner.txt"
        rm -f "$mine"
        echo "[gpu_lock] 잡음: $owner / $purpose ($minutes 분)"
        return 0
      fi
      if _gl_epoch_stale "$GPU_LOCK_DIR/owner.txt"; then
        echo "[gpu_lock] 오래된 잠금 지움(end_epoch + 900 초 지남): $(tr '\n' ' ' < "$GPU_LOCK_DIR/owner.txt")"
        rm -rf "$GPU_LOCK_DIR"
        continue
      fi
    fi
    if [ $(($(_gl_now) - t0)) -gt $((60 * maxwait)) ]; then
      rm -f "$mine"
      echo "[gpu_lock] $maxwait 분 기다려도 못 잡음"
      return 1
    fi
    o=$(_gl_key "$GPU_LOCK_DIR/owner.txt" owner)
    left=$(_gl_key "$GPU_LOCK_DIR/owner.txt" end_epoch)
    [[ "$left" =~ ^[0-9]+$ ]] && left="끝 예정까지 $(((left - $(_gl_now)) / 60)) 분" || left=
    ahead=
    for ((i = 0; i < pos; i++)); do ahead+="${ahead:+, }$(basename "${q[$i]}" | sed 's/^[0-9]*_//')"; done
    if [ -n "$o" ]; then held="$o / $(_gl_key "$GPU_LOCK_DIR/owner.txt" purpose) / $left"
    elif [ -d "$GPU_LOCK_DIR" ]; then held='(owner.txt 없음)'
    else held='없음'; fi
    echo "[gpu_lock] 대기열 $((pos + 1))/${#q[@]} (앞: $ahead) · 잠금: $held -- ${GPU_LOCK_POLL} 초 뒤 다시"
    sleep "$GPU_LOCK_POLL"
  done
}

gpu_lock_cancel() {  # 기다리다 그만둘 때(신호 처리 등): 지금 기다리는 내 표만 지운다
  [ -n "${_GL_MINE:-}" ] && rm -f "$_GL_MINE" "$_GL_MINE".tmp* 2>/dev/null
  return 0
}

gpu_lock_exit() {  # gpu_lock_exit <owner> : 내 잠금만 푼다. owner.txt 가 없으면(형식이 다른 남의 잠금일 수 있음) 안 건드린다
  local owner=$1 o
  o=$(_gl_key "$GPU_LOCK_DIR/owner.txt" owner)
  if [ -z "$o" ] || [ "$o" != "$owner" ]; then
    [ -d "$GPU_LOCK_DIR" ] && echo "[gpu_lock] 내 잠금이 아니라 안 푼다: ${o:-?}"
    return 1
  fi
  rm -rf "$GPU_LOCK_DIR"
  echo "[gpu_lock] 풂: $owner"
  return 0
}

gpu_lock_status() {
  local t
  if [ -f "$GPU_LOCK_DIR/owner.txt" ]; then
    echo '[잠금]'; tr -d '\r' < "$GPU_LOCK_DIR/owner.txt" | grep -E '^[A-Za-z_]+=' | sort | sed 's/^/  /'
  elif [ -d "$GPU_LOCK_DIR" ]; then echo '[잠금] 잠김(owner.txt 없음 또는 형식 다름)'
  else echo '[잠금] free'; fi
  echo "[대기열] now_epoch=$(_gl_now)"
  while read -r t; do
    [ -n "$t" ] || continue
    echo "  $(basename "$t")  $(_gl_key "$t" purpose) / end_epoch=$(_gl_key "$t" end_epoch) / host=$(_gl_key "$t" host) pid=$(_gl_key "$t" pid)"
  done < <(_gl_tickets)
}

_gl_usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

_gl_main() {
  set -euo pipefail
  local action=${1:-} owner=fasteval purpose='' minutes=30 vram='?' maxwait=120
  [ $# -gt 0 ] && shift
  while [ $# -gt 0 ]; do
    case $1 in
      --owner) owner=$2; shift 2 ;;
      --purpose) purpose=$2; shift 2 ;;
      --minutes) minutes=$2; shift 2 ;;
      --vram-gb) vram=$2; shift 2 ;;
      --max-wait-min) maxwait=$2; shift 2 ;;
      -h|--help) _gl_usage; return 0 ;;
      *) echo "모르는 인자: $1" >&2; _gl_usage >&2; return 2 ;;
    esac
  done
  case $action in
    acquire) gpu_lock_enter "$owner" "$purpose" "$minutes" "$vram" "$maxwait" ;;
    release) gpu_lock_exit "$owner" ;;
    status) gpu_lock_status ;;
    -h|--help|'') _gl_usage ;;
    *) echo "모르는 명령: $action (acquire|release|status)" >&2; return 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  # 기다리다 Ctrl-C 로 끊으면 내 표를 지운다
  trap 'gpu_lock_cancel; exit 130' INT TERM
  _gl_main "$@"
fi
