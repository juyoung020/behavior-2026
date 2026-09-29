"""가속 파이프라인이 원래 openpi 파이프라인과 같은 텐서를 내는지 대조한다 (JSBSim fdm_verify --check 와 같은 틀).

판정: "비트 동일" / "허용오차 안 (최대 |오차|)" / "다름". 기준은 원래 파이프라인이 같은 인덱스에서 낸 텐서다.

    bash tools/ft_run.sh tools/ft_verify.py index                 # Rust mp4 색인 == libavformat(PyAV) 패킷 (GPU 안 씀)
    bash tools/ft_run.sh tools/ft_verify.py stage                 # 구간별: NVDEC+색표 == torchcodec RGB, 커널 == JAX 크기 조정
    bash tools/ft_run.sh tools/ft_verify.py ref --tag a           # 원래 파이프라인 기준 저장 (고정 인덱스)
    bash tools/ft_run.sh tools/ft_verify.py ref --tag b           # 다른 프로세스에서 한 번 더
    bash tools/ft_run.sh tools/ft_verify.py same a b              # 원래가 매번 같은가
    bash tools/ft_run.sh tools/ft_verify.py check --tag a         # 가속판 vs 기준 (다르면 exit 1)
    bash tools/ft_run.sh tools/ft_verify.py check --tag a --negative lut   # 음성 대조: 일부러 틀리게 해서 잡는지
    bash tools/ft_run.sh tools/ft_verify.py loader --batches 3    # 로더 통째 (섞기 순서·묶기·GPU 전송·uint8→float 까지)

무작위 증강: 데이터 파이프라인에는 없다(증강은 학습 스텝 안 model.preprocess_observation). 섞기 순서는 seed 42 고정
(openpi data_loader.py:488-489) — loader 대조는 두 로더를 같은 seed 로 만든다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
WORK = os.environ.get("FT_WORK", os.path.expanduser("~/fasttrain_work"))
REF = os.path.join(WORK, "ref")


def flat(x, prefix=""):
    """중첩 dict → {'image/base_0_rgb': arr, ...}"""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}{k}/"))
    else:
        out[prefix[:-1]] = np.asarray(x)
    return out


def verdict(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape or a.dtype != b.dtype:
        return "다름", f"모양/형 {a.shape}{a.dtype} vs {b.shape}{b.dtype}", np.inf
    if np.array_equal(a, b):
        return "비트 동일", "", 0.0
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    n = int((a != b).sum())
    return ("허용오차 안" if d.max() <= 1e-6 * max(1.0, np.abs(b).max()) else "다름"), f"다른 값 {n}/{a.size}", float(d.max())


def fixed_indices(lds, n_random: int, seed: int = 1234):
    """무작위 + 경계 사례(에피소드 처음·끝(행동 패딩), GOP 경계, 파일이 바뀌는 에피소드)."""
    meta = lds.meta
    rng = np.random.default_rng(seed)
    idx = list(rng.integers(0, len(lds), n_random))
    eps = sorted(lds.episodes)
    starts = {}
    pos = 0
    for e in eps:  # 선택된 에피소드 순서대로 이어 붙은 상대 인덱스
        L = meta.episodes[e]["length"]
        starts[e] = (pos, L)
        pos += L
    edge_eps = [eps[0], eps[1], eps[-1]]
    key = "videos/observation.rgb.zed_link_camera_0/file_index"
    for a, b in zip(eps[:-1], eps[1:]):
        if meta.episodes[a][key] != meta.episodes[b][key]:
            edge_eps.append(b)  # 머리 카메라 파일이 바뀌는 첫 에피소드
    for e in edge_eps:
        s, L = starts[e]
        idx += [s, s + 1, s + 7, s + 8, s + L - 33, s + L - 32, s + L - 2, s + L - 1]
    return [int(i) for i in idx]


# ------------------------------------------------------------------ Rust 색인
def cmd_index(a):
    """Rust ftprep 가 mp4 에서 읽은 패킷(위치·크기·키프레임·표시 시각)이 libavformat 이 읽은 것과 같은지. GPU 안 씀."""
    import glob

    import av
    from fasttrain import fast

    root = os.path.expanduser("~/data/2026-challenge-demos/videos")
    bad_total, n_total = 0, 0
    for v in sorted(glob.glob(f"{root}/observation.rgb.*/chunk-*/*.mp4")):
        b = np.fromfile(fast.ensure_index(v), np.uint8)
        _, _, _, _, plen, _ = np.frombuffer(b[8:32].tobytes(), "<u4")
        n = int(np.frombuffer(b[32 + plen: 40 + plen].tobytes(), "<u8")[0])
        rec = np.frombuffer(b[40 + plen: 40 + plen + 24 * n].tobytes(), fast._IDX_REC)
        cols = {"pos": [], "size": [], "key": [], "pts": []}
        with av.open(v) as c:
            for p in c.demux(c.streams.video[0]):
                if p.size:
                    cols["pos"].append(p.pos), cols["size"].append(p.size)
                    cols["key"].append(int(p.is_keyframe)), cols["pts"].append(p.pts)
        cols = {k: np.asarray(x) for k, x in cols.items()}
        same_n = len(cols["pos"]) == n
        bad = {k: int((cols[k] != rec[f]).sum()) if same_n else -1
               for k, f in (("pos", "off"), ("size", "size"), ("key", "key"), ("pts", "pts"))}
        bad_total += (not same_n) + sum(max(0, x) for x in bad.values())
        n_total += n
        print(f"  {v.split('videos/')[1]:<62} 패킷 {n:>7} (PyAV {len(cols['pos']):>7})  다른 것 {bad}")
    print(f"패킷 {n_total} 개: " + ("통과" if bad_total == 0 else f"실패 {bad_total}"))
    return 1 if bad_total else 0


# ------------------------------------------------------------------ 구간별
def cmd_stage(a):
    import torch
    from openpi.shared import image_tools
    from fasttrain import fast, lut

    root = os.path.expanduser("~/data/2026-challenge-demos/videos")
    cams = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
            "observation.rgb.right_realsense_link_camera_0"]
    videos = [f"{root}/{c}/chunk-000/file-00{k}.mp4" for c in cams for k in (0, 1)]
    idxs = [fast.ensure_index(v) for v in videos]
    eng = fast.make_engine(videos, idxs, threads=4)
    print(f"크기 조정 보정 (이 GPU 의 원래 JAX 크기 조정이 나누어 더하는 경계, 입력 좌표): {eng.calib}")
    print(f"보정 후 uint8 무작위 영상 24장×2 크기: 커널 != 원래 인 값 {eng.validated}")
    rng = np.random.default_rng(a.seed)
    print("=" * 78)
    print(f"구간별 대조 (파일 {len(videos)}개 × 프레임 {a.n}개, 무작위 seed={a.seed})")
    print("=" * 78)
    print(f"  {'파일':<52}{'RGB(색표)':>12}{'224²':>12}")
    bad = 0
    for f, v in enumerate(videos):
        W, H, ts, nfr = eng.info(f)
        frames = [int(x) for x in rng.integers(0, nfr, a.n)]
        frames[0] = 0
        frames[-1] = nfr - 1
        req = torch.tensor([[f, t] for t in frames], dtype=torch.int64)
        rgb = torch.empty((a.n, H, W, 3), dtype=torch.uint8, device="cuda")
        eng.run(req, rgb, 1)
        small = torch.empty((a.n, 224, 224, 3), dtype=torch.uint8, device="cuda")
        eng.run(req, small, 0)
        ref_rgb = lut.original_rgb(v, frames)  # 원래: torchcodec → /255 → (255*x).astype(uint8)
        # 원래: JAX GPU 크기 조정을 **한 장씩** (ResizeImages 가 샘플마다 부른다 — 여러 장을 묶으면 GEMM 모양이 달라져 비트가 바뀐다)
        ref_small = np.stack([np.asarray(image_tools.resize_with_pad(x, 224, 224)) for x in ref_rgb])
        g_rgb, g_small = rgb.cpu().numpy(), small.cpu().numpy()
        n1 = int((g_rgb != ref_rgb).sum())
        n2 = int((g_small != ref_small).sum())
        bad += n1 + n2
        print(f"  {v.split('videos/')[1]:<52}{n1:>12}{n2:>12}")
    print(f"\n  (표의 숫자 = 원래와 다른 값의 개수, 0 이면 비트 동일)   엔진 통계 {eng.stats()}")
    print("통과" if bad == 0 else f"실패 — 다른 값 {bad}")
    return 1 if bad else 0


# ------------------------------------------------------------------ 기준 저장·자기 일치
def cmd_ref(a):
    from fasttrain import orig

    cfg = orig.train_config()
    ds, dc = orig.dataset(cfg)
    lds = orig.lerobot_of(ds)
    idx = fixed_indices(lds, a.n)
    os.makedirs(REF, exist_ok=True)
    t = time.time()
    cols = {}
    for i in idx:
        for k, v in flat(ds[i]).items():
            cols.setdefault(k, []).append(v)
    arrs = {k: np.stack(v) for k, v in cols.items()}
    arrs["_idx"] = np.asarray(idx)
    path = os.path.join(REF, f"ref_{a.tag}.npz")
    np.savez(path, **arrs)
    print(f"원래 파이프라인 기준 {len(idx)}개 샘플 → {path} ({time.time()-t:.0f}s)")
    for k, v in arrs.items():
        print(f"  {k:<32}{str(v.shape):<24}{v.dtype}")


def cmd_same(a):
    A = np.load(os.path.join(REF, f"ref_{a.a}.npz"))
    B = np.load(os.path.join(REF, f"ref_{a.b}.npz"))
    print(f"원래 파이프라인 자기 일치: {a.a} vs {a.b} (다른 프로세스)")
    bad = 0
    for k in A.files:
        v, why, err = verdict(A[k], B[k])
        bad += v != "비트 동일"
        print(f"  {k:<32}{v:<12}{why}")
    return 1 if bad else 0


# ------------------------------------------------------------------ 가속판 대조
def fast_samples(idx, negative: str = ""):
    import torch
    from fasttrain import fast, orig

    cfg = orig.train_config()
    fds = fast.FastDataset(cfg)
    lutt, hook = None, None
    if negative == "lut":  # 음성 대조 1: 색표 대신 부동소수 BT.601 공식 (최대 3 차이 나는 흔한 구현)
        k = np.arange(1 << 24)
        y = (k >> 16).astype(np.float64) - 16
        u = ((k >> 8) & 255).astype(np.float64) - 128
        v = (k & 255).astype(np.float64) - 128
        rgb = np.stack([1.164383 * y + 1.596027 * v, 1.164383 * y - 0.391762 * u - 0.812968 * v,
                        1.164383 * y + 2.017232 * u], -1)
        lutt = torch.from_numpy(np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(-1))
    if negative == "weight":  # 음성 대조 2: 크기 조정 가중치를 float32 한 칸(1 ulp)만 틀리게
        hook = lambda wt: np.where(wt != 0, np.nextafter(wt, np.float32(1)), wt).astype(np.float32)
    # 음성 대조 3 "frame": 프레임 번호를 하나 밀기 (시각 맞추기 실수) — 아래에서
    eng = fast.make_engine(fds.videos, fds.indexes, threads=6, lut_override=lutt, weight_hook=hook,
                           check=not negative)
    names = list(fast.NAMES)
    cols = {}
    t = time.time()
    xs = [fds[i] for i in idx]
    req = np.stack([x.pop("_ft_req") for x in xs])  # (N,3,2)
    if negative == "frame":
        req[:, :, 1] = np.maximum(req[:, :, 1] - 1, 0)
    out = torch.empty((len(idx) * 3, 224, 224, 3), dtype=torch.uint8, device="cuda")
    eng.run(torch.from_numpy(np.ascontiguousarray(req.reshape(-1, 2))), out, 0)
    imgs = out.view(len(idx), 3, 224, 224, 3).cpu().numpy()
    for j, x in enumerate(xs):
        for c, nm in enumerate(names):
            x["image"][nm] = imgs[j, c]
        for k, v in flat(x).items():
            cols.setdefault(k, []).append(v)
    print(f"가속판 {len(idx)}개 샘플 {time.time()-t:.1f}s, 엔진 통계 {eng.stats()}")
    return {k: np.stack(v) for k, v in cols.items()}


def cmd_check(a):
    R = np.load(os.path.join(REF, f"ref_{a.tag}.npz"))
    idx = [int(i) for i in R["_idx"]]
    F = fast_samples(idx, a.negative)
    print("=" * 78)
    print(f"가속판 vs 원래 기준 '{a.tag}' ({len(idx)} 샘플{', 음성 대조 ' + a.negative if a.negative else ''})")
    print("=" * 78)
    print(f"  {'텐서':<34}{'판정':<12}{'최대 |오차|':>12}   비고")
    bad = 0
    for k in R.files:
        if k == "_idx":
            continue
        if k not in F:
            print(f"  {k:<34}{'다름':<12}{'':>12}   가속판에 없음")
            bad += 1
            continue
        v, why, err = verdict(F[k], R[k])
        bad += v != "비트 동일"
        print(f"  {k:<34}{v:<12}{err:>12.3g}   {why}")
    extra = [k for k in F if k not in R.files]
    if extra:
        print(f"  가속판에만 있는 키: {extra}")
        bad += 1
    if a.negative:
        print(f"\n음성 대조: " + ("잡았다 (정상)" if bad else "못 잡았다 — 대조가 무디다"))
        return 0 if bad else 1
    print("\n" + ("통과 — 전부 비트 동일" if not bad else f"실패 — {bad} 개 텐서가 비트 동일이 아님"))
    return 1 if bad else 0


def cmd_loader(a):
    """두 로더를 같은 seed 로 만들어 배치를 통째로 (GPU 위 Observation 까지) 대조."""
    import jax
    from fasttrain import fast, orig

    cfg = orig.train_config(num_workers=a.workers, batch_size=a.batch)
    it_o = iter(orig.loader(cfg, shuffle=True, num_batches=a.batches))
    it_f = iter(fast.FastLoader(cfg, shuffle=True, num_batches=a.batches, num_workers=a.workers,
                                batch_size=a.batch, decode_threads=6))
    bad = 0
    for b in range(a.batches):
        (oo, oa), (fo, fa) = next(it_o), next(it_f)
        lo = flat({"obs": oo.to_dict(), "actions": oa})
        lf = flat({"obs": fo.to_dict(), "actions": fa})
        for k in lo:
            if lo[k].dtype == object or lo[k].shape == ():
                continue
            v, why, err = verdict(np.asarray(jax.device_get(lf[k])), np.asarray(jax.device_get(lo[k])))
            bad += v != "비트 동일"
            if b == 0 or v != "비트 동일":
                print(f"  배치 {b} {k:<40}{v:<12}{why}")
    print("\n" + ("통과 — 로더 출력 전부 비트 동일" if not bad else f"실패 — {bad}"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("index")
    s = sp.add_parser("stage")
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("ref")
    s.add_argument("--tag", default="a")
    s.add_argument("--n", type=int, default=224)
    s = sp.add_parser("same")
    s.add_argument("a")
    s.add_argument("b")
    s = sp.add_parser("check")
    s.add_argument("--tag", default="a")
    s.add_argument("--negative", default="", choices=["", "lut", "weight", "frame"])
    s = sp.add_parser("loader")
    s.add_argument("--batches", type=int, default=3)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    sys.exit({"index": cmd_index, "stage": cmd_stage, "ref": cmd_ref, "same": cmd_same, "check": cmd_check,
              "loader": cmd_loader}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
