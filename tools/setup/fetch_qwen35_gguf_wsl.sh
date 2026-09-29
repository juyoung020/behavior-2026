#!/usr/bin/env bash
# Qwen3.5-9B 양자화판(GGUF)을 WSL ~/models/ 에 받는다.
# - Qwen 공식 GGUF 는 없다(Qwen/Qwen3.5-9B-GGUF, ggml-org 모두 없음; 2026-09-29 HF API). unsloth 판 사용(Apache-2.0, base_model=Qwen/Qwen3.5-9B).
# - 받는 파일은 정확히 지정(HF 429 방지): 본체 Q4_K_M + 이미지 인코더 mmproj-F16. 인자로 다른 양자화 이름을 줄 수 있다.
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/setup/fetch_qwen35_gguf_wsl.sh [REPO] [QUANT_FILE]
set -euo pipefail
REPO=${1:-unsloth/Qwen3.5-9B-GGUF}
QFILE=${2:-Qwen3.5-9B-Q4_K_M.gguf}
DIR=~/models/$(basename "$REPO")
mkdir -p "$DIR"
cd "$DIR"
for f in "$QFILE" mmproj-F16.gguf; do
  echo "== $f"
  curl -L --fail --retry 5 -C - -o "$f" "https://huggingface.co/$REPO/resolve/main/$f"
done
# HF 가 알려주는 sha256 과 대조
curl -s "https://huggingface.co/api/models/$REPO/tree/main" | python3 -c "
import sys, json, hashlib, os
for e in json.load(sys.stdin):
    p = e['path']
    if p in ('$QFILE', 'mmproj-F16.gguf'):
        want = e.get('lfs', {}).get('oid')
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h.update(chunk)
        print(p, os.path.getsize(p), 'OK' if h.hexdigest() == want else 'MISMATCH', want)
"
ls -la "$DIR"
