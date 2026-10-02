"""224 정책 입력 영상 비교: 포팅 평가기(엔진 렌더)의 trace_images.npz vs 공식 평가기의 trace_images.npz (같은 행동열).

    python3 ssim224.py <공식 결과 폴더>[,<공식2>...] <우리 결과 폴더> [--csv out.csv]

밝기 SSIM(가우스 11, σ 1.5, Y = 0.299R + 0.587G + 0.114B, 0~255), 평균 |차|, 채널 평균 차. 공식이 검은 프레임(RGB 평균 < 2)이면 뺀다.
공식이 여러 개면 공식끼리의 SSIM(같은 설정 잡음 폭)도 같이 적는다.
"""
import argparse
import os
import sys

import numpy as np


def _gauss(sig=1.5, n=11):
    x = np.arange(n) - (n - 1) / 2
    g = np.exp(-(x ** 2) / (2 * sig * sig))
    return g / g.sum()


def _filt(a, g):
    from numpy.lib.stride_tricks import sliding_window_view

    p = len(g) // 2
    a = np.pad(a, p, mode="reflect")
    a = (sliding_window_view(a, len(g), axis=1) * g).sum(-1)
    a = (sliding_window_view(a, len(g), axis=0) * g).sum(-1)
    return a


def ssim(a, b):
    ya = (a[..., :3].astype(np.float64) @ [0.299, 0.587, 0.114])
    yb = (b[..., :3].astype(np.float64) @ [0.299, 0.587, 0.114])
    g = _gauss()
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _filt(ya, g), _filt(yb, g)
    va = _filt(ya * ya, g) - ma * ma
    vb = _filt(yb * yb, g) - mb * mb
    cab = _filt(ya * yb, g) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official")
    ap.add_argument("ours")
    ap.add_argument("--csv")
    a = ap.parse_args()
    offs = [np.load(os.path.join(d, "trace_images.npz")) for d in a.official.split(",")]
    ours = np.load(os.path.join(a.ours, "trace_images.npz"))
    rows = []
    cams = {"zed_link": "머리", "left_realsense_link": "왼손목", "right_realsense_link": "오른손목"}
    for k in sorted(ours.files, key=lambda s: (int(s.split("|")[0]), s)):
        step = int(k.split("|")[0])
        cam = next((v for c, v in cams.items() if c in k), k)
        o = ours[k]
        for j, off in enumerate(offs):
            if k not in off.files:
                continue
            f = off[k]
            if f[..., :3].mean() < 2:
                rows.append((step, cam, j, "공식 검음", None, None))
                continue
            rows.append((step, cam, j, "우리", ssim(f, o), float(np.abs(f[..., :3].astype(float) - o[..., :3]).mean())))
        if len(offs) > 1 and all(k in off.files for off in offs[:2]):
            f0, f1 = offs[0][k], offs[1][k]
            if f0[..., :3].mean() >= 2 and f1[..., :3].mean() >= 2:
                rows.append((step, cam, -1, "공식끼리", ssim(f0, f1), float(np.abs(f0[..., :3].astype(float) - f1[..., :3]).mean())))
    print(f"{'스텝':>4} {'카메라':<6} {'쌍':<8} {'SSIM':>7} {'|평균차|':>8}")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:<6} {r[3]:<8} " + ("   -   " if r[4] is None else f"{r[4]:7.3f}") + ("" if r[5] is None else f" {r[5]:8.2f}"))
    for cam in cams.values():
        v = [r[4] for r in rows if r[1] == cam and r[3] == "우리" and r[4] is not None]
        n = [r[4] for r in rows if r[1] == cam and r[3] == "공식끼리" and r[4] is not None]
        if v:
            print(f"요약 {cam}: 우리 SSIM 평균 {np.mean(v):.3f} (최소 {np.min(v):.3f}, {len(v)} 장)" +
                  (f" | 공식끼리 {np.mean(n):.3f}" if n else ""))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("step,cam,official_idx,pair,ssim,mean_abs\n")
            for r in rows:
                f.write(",".join("" if x is None else str(x) for x in r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
