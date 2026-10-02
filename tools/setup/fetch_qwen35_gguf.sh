#!/usr/bin/env bash
# Qwen3.5-9B 양자화판(GGUF)을 ~/models/ 에 받는다.
# (리눅스판; 옛 WSL 판 archive/tools/setup/fetch_qwen35_gguf_wsl.sh 와 같은 저장소·파일·위치·sha256 대조)
# - Qwen 공식 GGUF 는 없다(Qwen/Qwen3.5-9B-GGUF, ggml-org 모두 없음; 2026-09-29 HF API). unsloth 판 사용(Apache-2.0, base_model=Qwen/Qwen3.5-9B).
# - 받는 파일은 정확히 지정(HF 429 방지): 본체 Q4_K_M + 이미지 인코더 mmproj-F16. 인자로 다른 양자화 이름을 줄 수 있다.
#   bash tools/setup/fetch_qwen35_gguf.sh [--dry-run] [REPO] [QUANT_FILE]
#   --dry-run : 받지 않고 원격 크기·sha256 과 이미 받은 파일 상태만 찍는다
set -euo pipefail
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
case ${1:-} in -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
REPO=${1:-unsloth/Qwen3.5-9B-GGUF}
QFILE=${2:-Qwen3.5-9B-Q4_K_M.gguf}
DIR=~/models/$(basename "$REPO")
FILES=("$QFILE" mmproj-F16.gguf)
if [ $DRY = 1 ]; then
  echo "(dry-run) $REPO -> $DIR : ${FILES[*]}"
  curl -sf "https://huggingface.co/api/models/$REPO/tree/main" | DIR=$DIR QFILE=$QFILE python3 -c "
import sys, json, os
d = os.environ['DIR']
for e in json.load(sys.stdin):
    p = e['path']
    if p in (os.environ['QFILE'], 'mmproj-F16.gguf'):
        lp = os.path.join(d, p)
        have = os.path.getsize(lp) if os.path.exists(lp) else 0
        print(f\"  {p}: 원격 {e.get('size', 0) / 1e9:.2f} GB sha256 {e.get('lfs', {}).get('oid')} / 받아 둠 {have / 1e9:.2f} GB\")
"
  df -h "$HOME" | tail -1
  exit 0
fi
mkdir -p "$DIR"
cd "$DIR"
for f in "${FILES[@]}"; do
  echo "== $f"
  curl -L --fail --retry 5 -C - -o "$f" "https://huggingface.co/$REPO/resolve/main/$f"
done
# HF 가 알려주는 sha256 과 대조
curl -s "https://huggingface.co/api/models/$REPO/tree/main" | QFILE=$QFILE python3 -c "
import sys, json, hashlib, os
for e in json.load(sys.stdin):
    p = e['path']
    if p in (os.environ['QFILE'], 'mmproj-F16.gguf'):
        want = e.get('lfs', {}).get('oid')
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h.update(chunk)
        print(p, os.path.getsize(p), 'OK' if h.hexdigest() == want else 'MISMATCH', want)
"
ls -la "$DIR"
