#!/usr/bin/env bash
# 리눅스에서 검은 프레임이 나오는지 확인하는 최소 재현 (설치 -> 공식 평가기 그대로 -> 출력 MP4 에서 검은 칸 세기).
# 목적: 이 Windows PC(RTX 5070 Ti, 드라이버 591.86, Isaac Sim 5.1)에서 카메라 RGB 가 17~100% 검게 나오는 문제가
#       Windows 전용인지 가르기. 우리 코드·수정은 하나도 안 쓴다 (공식 r1pro.yaml, 공식 --policy local = 0 행동).
# 필요: Ubuntu 22.04/24.04, NVIDIA RTX GPU(RT 코어), 드라이버(Isaac Sim 5.1 시험값 Linux 580.65.06), 디스크 약 80 GB, conda.
# 실행: bash linux_black_frame_repro.sh [작업폴더(기본 ~/b1k_repro)]
#   이미 설치돼 있으면 설치 단계는 건너뛴다(BEHAVIOR-1K 폴더와 conda 환경 behavior 가 있으면).
# 결과: <작업폴더>/out_*/videos/*.mp4 와 검은 칸 개수. 한 장이라도 검으면 exit 1.
set -euo pipefail
WORK=${1:-$HOME/b1k_repro}
TAG=v3.9.3-post1                         # 대회 평가 태그 (docs/raw/site_challenge_evaluation.md "Use the v3.9.3 tag")
mkdir -p "$WORK"; cd "$WORK"

echo "== 환경"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
lsb_release -ds 2>/dev/null || cat /etc/os-release | head -2

# 1) 설치 (공식 setup.sh 그대로)
if [ ! -d BEHAVIOR-1K ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/StanfordVL/BEHAVIOR-1K.git
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^behavior\s'; then
  (cd BEHAVIOR-1K && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval \
      --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos)
fi
conda activate behavior
python -c "import isaacsim, os; print('Isaac Sim', open(os.path.join(os.path.dirname(isaacsim.__file__),'VERSION')).read().strip())"

# 2) 공식 평가기 그대로: radio 인스턴스 0, 헤드리스, 영상 저장, 공식 로봇 설정(--robot-config 없음), 0 행동 정책
run() {  # $1 = 이름, 나머지 = 추가 인자
  local name=$1; shift
  local out="$WORK/out_$name"
  rm -rf "$out"
  (cd BEHAVIOR-1K/OmniGibson && OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.eval.eval \
      --task-name turning_on_radio --mode public_test --instance-indices 0 --max-steps 300 \
      --policy local --output-dir "$out" --write-video --headless "$@") 2>&1 | tail -3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
}
run n1                                              # 환경 1 개
run n2 --instance-indices 0 1 --num-envs 2          # 공식 --num-envs 2

# 3) 출력 MP4 를 디코딩해 카메라 칸별로 검은 프레임을 센다 (평가기 영상 = 정책 관측: 왼손목 위, 오른손목 아래, 머리 오른쪽)
python - "$WORK" <<'PY'
import glob, sys
import av
bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/out_*/videos/*.mp4")):
    n, black = 0, [0, 0, 0]
    with av.open(f) as c:
        for fr in c.decode(video=0):
            a = fr.to_ndarray(format="rgb24"); n += 1
            for i, part in enumerate((a[:224, :224], a[224:, :224], a[:, 224:])):
                black[i] += part.mean() < 2
    bad += sum(black)
    print(f"{f}: {n} frames, black left_wrist={black[0]} right_wrist={black[1]} head={black[2]}")
print("RESULT:", "black frames found" if bad else "no black frames")
sys.exit(1 if bad else 0)
PY
