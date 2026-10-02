#!/usr/bin/env bash
# 2025 상위 팀 공개 체크포인트(추론용만)를 ~/checkpoints 에 받는다. 안 A = 1위 제출 4개 + Comet pt50, 약 63 GB.
# (리눅스판; 옛 WSL 판 archive/tools/setup/download_top_ckpts_wsl.sh 와 같은 저장소·파일·위치)
# 근거·크기: docs/2025상위팀_깃허브.md 1.2절. 끊겨도 다시 돌리면 이어받는다.
#   bash tools/setup/download_top_ckpts.sh            # 받기 (로그: logs/download_top_ckpts_<시각>.log)
#   bash tools/setup/download_top_ckpts.sh --dry-run  # 받지 않고 원격 파일 목록·크기와 이미 받은 것을 비교만
# 파이썬: HF_PY(기본 ~/openpi/.venv/bin/python, huggingface_hub 필요). HF 토큰은 huggingface_hub 가 알아서 읽는다(찍지 않음).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DRY=0
case ${1:-} in --dry-run) DRY=1 ;; -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;; '') ;; *) echo "모르는 인자: $1" >&2; exit 2 ;; esac
PY=${HF_PY:-$HOME/openpi/.venv/bin/python}
mkdir -p "$REPO/logs" ~/checkpoints
LOG=$REPO/logs/download_top_ckpts_$(date +%Y%m%d_%H%M)$([ $DRY = 1 ] && echo _dry).log
DRY=$DRY "$PY" - <<'PY' 2>&1 | tee "$LOG"
import fnmatch
import os
from huggingface_hub import HfApi, snapshot_download
base = os.path.expanduser("~/checkpoints")
jobs = [
    ("IliaLarchenko/behavior_submission", "behavior_submission", dict(ignore_patterns=["replay.mp4"])),
    ("sunshk/openpi_comet", "openpi_comet", dict(allow_patterns=["pi05-b1kpt50-cs32/**"])),
]
dry = os.environ.get("DRY") == "1"
for repo, name, kw in jobs:
    print(f"== {repo} -> {base}/{name}", flush=True)
    if dry:
        info = HfApi().model_info(repo, files_metadata=True)
        tot = have = 0
        miss = []
        for s in info.siblings:
            f = s.rfilename
            if any(fnmatch.fnmatch(f, p) for p in kw.get("ignore_patterns", [])):
                continue
            if "allow_patterns" in kw and not any(fnmatch.fnmatch(f, p) for p in kw["allow_patterns"]):
                continue
            size = s.size or 0
            tot += size
            lp = os.path.join(base, name, f)
            if os.path.exists(lp) and os.path.getsize(lp) == size:
                have += size
            else:
                miss.append((f, size))
        print(f"   원격 {tot / 1e9:.2f} GB, 받아 둠(크기 같음) {have / 1e9:.2f} GB, 받을 것 {len(miss)} 개 {(tot - have) / 1e9:.2f} GB", flush=True)
        for f, size in miss[:20]:
            print(f"   - {f} ({size / 1e9:.2f} GB)", flush=True)
        continue
    snapshot_download(repo, local_dir=f"{base}/{name}", max_workers=8, **kw)
    print(f"== {repo} 완료", flush=True)
print("=== DRY RUN DONE ===" if dry else "=== DOWNLOAD DONE ===", flush=True)
PY
du -sh ~/checkpoints/* | tee -a "$LOG"
