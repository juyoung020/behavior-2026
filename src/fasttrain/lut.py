"""색 변환 표(LUT) 만들기와 확인 — 파이썬은 원래 디코더(torchcodec)를 부르는 접착부만 한다.

원래 파이프라인에서 픽셀이 겪는 일 (영상 → 크기 조정 직전):
  1. torchcodec(CPU) 가 HEVC 를 풀어 YUV420 → RGB24 로 바꾼다 (FFmpeg swscale 의 고정소수점 경로).
     lerobot/datasets/video_utils.py:267-352
  2. lerobot 이 /255 해서 float32 로 (video_utils.py:350-352)
  3. openpi b1k_policy._parse_image 가 (255 * x).astype(uint8) 로 되돌린다 (b1k_policy.py:44-50)
실측(tools 의 확인 결과): 결과 RGB 는 (Y, 그 픽셀이 속한 2x2 블록의 U, V) 만의 함수다. 그래서 모든 (Y,U,V) 2^24 조합을
담은 합성 영상(Rust `ftprep yuvgrid`)을 무손실 HEVC 로 만들어 **원래 1~3 단계 그대로** 풀면, 그 결과가 곧 정확한 표다.
GPU 에서는 NVDEC 가 준 YUV 에 이 표를 적용한다 (csrc/ftcore.cu).

    bash tools/ft_run.sh src/fasttrain/lut.py build      # 표 만들기 (720 폭·480 폭 각각: 원래 경로가 폭에 따라 다를 수 있어서)
    bash tools/ft_run.sh src/fasttrain/lut.py check      # 실제 영상 프레임으로 확인
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

WORK = os.environ.get("FT_WORK", os.path.expanduser("~/fasttrain_work"))
LUT_DIR = os.path.join(WORK, "lut")
FTPREP = os.path.join(WORK, "target", "release", "ftprep")
TOTAL_BLOCKS = 1 << 22


def lut_path(width: int) -> str:
    return os.path.join(LUT_DIR, f"lut_w{width}.bin")


def original_rgb(path: str, frame_indices, fps: float = 30.0) -> np.ndarray:
    """원래 1~3 단계를 그대로: lerobot torchcodec 디코딩(float32 /255) → openpi (255*x).astype(uint8). (N,H,W,3)"""
    from lerobot.datasets.video_utils import decode_video_frames_torchcodec

    t = decode_video_frames_torchcodec(path, [i / fps for i in frame_indices], 1e-3)  # float32 (N,3,H,W)
    img = (255 * np.asarray(t)).astype(np.uint8)  # b1k_policy._parse_image 와 같은 식
    return img.transpose(0, 2, 3, 1)


def build(width: int):
    os.makedirs(LUT_DIR, exist_ok=True)
    h = width
    yuv = os.path.join(LUT_DIR, f"grid_{width}.yuv")
    mp4 = os.path.join(LUT_DIR, f"grid_{width}.mp4")
    subprocess.run([FTPREP, "yuvgrid", str(width), str(h), yuv], check=True)
    per_frame = (width // 2) * (h // 2)
    nfr = (TOTAL_BLOCKS + per_frame - 1) // per_frame
    # 무손실 HEVC, 원본과 같은 yuv420p·제한 범위(tv)·색공간 미지정
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{h}",
                    "-r", "30", "-i", yuv, "-c:v", "libx265", "-x265-params", "lossless=1:log-level=0",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", mp4], check=True)
    os.remove(yuv)
    lut = np.zeros((1 << 24) * 3, np.uint8).reshape(-1, 3)
    seen = np.zeros(1 << 24, bool)
    bw = width // 2
    by, bx = np.divmod(np.arange(per_frame), bw)
    for f0 in range(0, nfr, 8):
        idx = list(range(f0, min(nfr, f0 + 8)))
        rgb = original_rgb(mp4, idx)
        for k, fi in enumerate(idx):
            c = fi * per_frame + np.arange(per_frame)
            ok = c < TOTAL_BLOCKS
            c = c[ok]
            U = (c >> 14) & 255
            V = (c >> 6) & 255
            yb = (c & 63) * 4
            for dy in range(2):
                for dx in range(2):
                    Y = yb + dy * 2 + dx
                    key = (Y << 16) | (U << 8) | V
                    lut[key] = rgb[k, 2 * by[ok] + dy, 2 * bx[ok] + dx]
                    seen[key] = True
    assert seen.all(), f"표가 다 안 찼다: {(~seen).sum()}"
    lut.tofile(lut_path(width))
    os.remove(mp4)
    print(f"표 {lut_path(width)} ({lut.nbytes/1e6:.0f} MB)")
    return lut


def yuv_planes(path: str, frame_index: int):
    """PyAV(다른 FFmpeg 빌드)로 같은 프레임의 YUV 평면. HEVC 디코딩은 규격상 비트 단위로 정해져 있다."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        tb = s.time_base
        c.seek(int(frame_index / 30.0 / tb), stream=s, backward=True)
        for fr in c.decode(s):
            if round(float(fr.pts * tb) * 30) == frame_index:
                a = fr.to_ndarray()
                H = fr.height
                W = fr.width
                return a[:H], a[H:H + H // 4].reshape(H // 2, W // 2), a[H + H // 4:].reshape(H // 2, W // 2)
    raise RuntimeError("프레임 못 찾음")


def apply(lut, Y, U, V):
    Un = np.repeat(np.repeat(U, 2, 0), 2, 1).astype(np.int64)
    Vn = np.repeat(np.repeat(V, 2, 0), 2, 1).astype(np.int64)
    return lut[(Y.astype(np.int64) << 16) | (Un << 8) | Vn]


def check(n_frames: int = 12, seed: int = 0):
    root = os.path.expanduser("~/data/2026-challenge-demos/videos")
    rng = np.random.default_rng(seed)
    bad_total = 0
    for cam, width in [("observation.rgb.zed_link_camera_0", 720), ("observation.rgb.left_realsense_link_camera_0", 480),
                       ("observation.rgb.right_realsense_link_camera_0", 480)]:
        lut = np.fromfile(lut_path(width), np.uint8).reshape(-1, 3)
        path = f"{root}/{cam}/chunk-000/file-000.mp4"
        cam_bad = 0
        for fi in rng.integers(0, 20000, n_frames):
            ref = original_rgb(path, [int(fi)])[0]
            got = apply(lut, *yuv_planes(path, int(fi)))
            cam_bad += int((ref != got).sum())
        bad_total += cam_bad
        print(f"{cam}: 프레임 {n_frames}개 ({ref.size * n_frames} 값), 표 결과 != 원래 결과 인 값 {cam_bad}")
    return bad_total


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        a = build(720)
        b = build(480)
        d = (a != b).any(1)
        print(f"720 폭 표 vs 480 폭 표: 다른 (Y,U,V) {int(d.sum())} / {1 << 24}")
    elif cmd == "check":
        sys.exit(1 if check() else 0)
