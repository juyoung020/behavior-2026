"""평가 결과 폴더에서 정책에 들어간 카메라 영상이 검었던(RGB 전부 0) 프레임 수를 센다 -- 계측 없이 돈 공식 실행 뒤 점검용.

    python C:\\behavior-2026\\tools\\black_frame_check.py <결과폴더> [<결과폴더> ...] [--max-ratio 0]

- videos\\*.mp4 : 평가기가 정책 관측으로 만든 영상(eval/evaluator.py _write_video: 왼손목·오른손목 224 세로, 머리 448)을 칸별로 본다.
  인코딩(x264) 때문에 완전 0 은 아니어서 칸 평균 < 2 를 검정으로 센다.
- server_log.npz(재생 서버) · black_frames.json(-BlackDiag/-BlackGuard) 가 있으면 그것도 같이 적는다(정확한 값).
- --max-ratio R : 어느 칸이든 검은 비율이 R 를 넘으면 exit 1 (기본 0 = 한 장이라도 있으면 실패).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np


def video_counts(path):
    import av

    n, black = 0, [0, 0, 0]
    with av.open(path) as c:
        for fr in c.decode(video=0):
            a = fr.to_ndarray(format="rgb24")
            n += 1
            for i, part in enumerate((a[:224, :224], a[224:, :224], a[:, 224:])):
                black[i] += part.mean() < 2
    return n, black


def server_counts(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        if not k.startswith("hash::") or not k.endswith("::rgb"):
            continue
        col = z[k]
        # 영(0) 배열 해시와 비교 -- 해상도마다 다르므로 흔한 크기 둘
        zeros = {hashlib.blake2b(np.zeros(s, np.uint8).tobytes(), digest_size=16).hexdigest()
                 for s in ((224, 224, 4), (480, 480, 4), (720, 720, 4))}
        out[k.split("::")[-2]] = (int(np.isin(col, list(zeros)).sum()), int(col.size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--max-ratio", type=float, default=0.0)
    a = ap.parse_args()
    worst = 0.0
    for d in a.dirs:
        print(f"== {d}")
        for v in sorted(glob.glob(os.path.join(d, "videos", "*.mp4"))):
            n, b = video_counts(v)
            worst = max(worst, max(b) / max(n, 1))
            print(f"  영상 {os.path.basename(v)}: {n} 프레임, 검은 칸 왼손목 {b[0]} · 오른손목 {b[1]} · 머리 {b[2]}")
        p = os.path.join(d, "server_log.npz")
        if os.path.exists(p):
            for cam, (k, n) in server_counts(p).items():
                print(f"  서버가 받은 관측 {cam}: 검정 {k}/{n}")
        p = os.path.join(d, "black_frames.json")
        if os.path.exists(p):
            j = json.load(open(p, encoding="utf-8"))
            print(f"  평가기 기록: 검정 {j.get('black')} / 전체 {j.get('total')}")
    ok = worst <= a.max_ratio
    print(f"가장 높은 검은 비율 {worst * 100:.1f}% -> {'통과' if ok else '실패'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
