"""네이티브 로더가 원래 openpi 파이프라인과 같은 텐서를 내는지 대조한다 (JSBSim fdm_verify --check 와 같은 틀).

판정: 텐서마다 "비트 동일" / "허용오차 안 (최대 |오차|)" / "다름". 기준은 원래 파이프라인이 같은 인덱스에서 낸 텐서를
JAX 에 넘어가는 형(float64→float32, int64→int32)으로 바꾼 것. 하나라도 비트 동일이 아니면 exit 1.
이 도구는 파이썬이다 — 기준 쪽이 원래 openpi·JAX·torchcodec(파이썬에만 있음)이라서. 대조 대상(네이티브)은 C ABI 로 부른다.

GPU 없이 (JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= bash tools/ft_run.sh tools/ft_verify.py ...):
    index                 Rust mp4 색인 == libavformat(PyAV) 패킷
    order                 네이티브 섞기 순서 == torch DataLoader (워커 지속·워커 0, 에포크 경계 넘어)
    table [--n N]         이미지 외 값(행동·상태·토큰·마스크) + 영상 프레임 번호, 과제 전체 샘플 (기본 전부)
GPU (bash tools/ft_run.sh tools/ft_verify.py ...):
    stage                 NVDEC+색표 == torchcodec RGB, 크기 조정 커널 == 원래 PIL (파일 6개 × 프레임 24)
    ref --tag a|b         원래 파이프라인 기준 저장 (고정 인덱스 288 = 무작위 224 + 경계 사례)
    same a b              원래가 두 프로세스에서 같은가
    check --tag a         네이티브 샘플 vs 기준 전체 텐서.  --negative lut|resize|frame 음성 대조
    loader --batches 3    로더 통째: 원래 로더 vs 네이티브 로더 (같은 seed, 섞기 순서·GPU 위 Observation 까지)
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
WORK = os.environ.get("FT_WORK", os.path.expanduser("~/fasttrain_work"))
REF = os.path.join(WORK, "ref")
FIELDS = ("actions", "state", "tokenized_prompt", "tokenized_prompt_mask", "image_mask")


def flat(x, prefix=""):
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


def table_files(tdir: str):
    files, idx = [], []
    for line in open(os.path.join(tdir, "table.txt")):
        if line.startswith("file\t"):
            _, v, i = line.rstrip("\n").split("\t")
            files.append(v), idx.append(i)
    return files, idx


def fixed_indices(lds, n_random: int, seed: int = 1234):
    """무작위 + 경계 사례(에피소드 처음·끝(행동 창이 잘리는 곳), GOP 경계, 머리 카메라 파일이 바뀌는 에피소드)."""
    meta = lds.meta
    rng = np.random.default_rng(seed)
    idx = list(rng.integers(0, len(lds), n_random))
    eps = sorted(lds.episodes)
    starts, pos = {}, 0
    for e in eps:
        L = meta.episodes[e]["length"]
        starts[e] = (pos, L)
        pos += L
    edge = [eps[0], eps[1], eps[-1]]
    key = "videos/observation.rgb.zed_link_camera_0/file_index"
    for a, b in zip(eps[:-1], eps[1:]):
        if meta.episodes[a][key] != meta.episodes[b][key]:
            edge.append(b)
    for e in edge:
        s, L = starts[e]
        idx += [s, s + 1, s + 7, s + 8, s + L - 33, s + L - 32, s + L - 2, s + L - 1]
    return [int(i) for i in idx]


# ------------------------------------------------------------------ Rust 색인 (GPU 없음)
def cmd_index(a):
    import glob

    import av
    from fasttrain import fast

    root = os.path.expanduser("~/data/2026-challenge-demos/videos")
    rec_t = np.dtype([("off", "<u8"), ("size", "<u4"), ("key", "<u4"), ("pts", "<i8")])
    bad_total, n_total = 0, 0
    for v in sorted(glob.glob(f"{root}/observation.rgb.*/chunk-*/*.mp4")):
        b = np.fromfile(fast.ensure_index(v), np.uint8)
        plen = int(np.frombuffer(b[24:28].tobytes(), "<u4")[0])
        n = int(np.frombuffer(b[32 + plen: 40 + plen].tobytes(), "<u8")[0])
        rec = np.frombuffer(b[40 + plen: 40 + plen + 24 * n].tobytes(), rec_t)
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


# ------------------------------------------------------------------ 섞기 순서 (GPU 없음)
class _Idx:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _ident(x):
    return x


def torch_order(n, bs, seed, workers, nbatches):
    """원래 openpi TorchDataLoader 와 같은 설정의 torch DataLoader 가 내는 인덱스 (끝나면 다시 iter, data_loader.py:508-519)."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(_Idx(n), batch_size=bs, shuffle=True, num_workers=workers,
                                     multiprocessing_context=multiprocessing.get_context("spawn") if workers else None,
                                     persistent_workers=workers > 0, collate_fn=_ident, drop_last=True, generator=g)
    out = []
    while len(out) < nbatches:
        for b in dl:
            out.append(list(b))
            if len(out) >= nbatches:
                break
    return np.asarray(out, np.int64)


def cmd_order(a):
    from fasttrain import fast

    bad = 0
    cases = [(429928, 32, 42, 2, 13435 * 2 + 5), (429928, 32, 42, 0, 13435 + 5), (1000, 32, 7, 2, 31 * 3 + 2),
             (1024, 32, 7, 2, 32 * 2 + 3), (1003, 16, 1, 0, 62 * 3)]
    print("섞기 순서: 네이티브(C++ MT19937) vs torch DataLoader")
    for n, bs, seed, w, nb in cases:
        t = torch_order(n, bs, seed, w, nb)
        f = fast.sampler_order(n, bs, seed, nb, shuffle=True, persistent=w > 0)
        same = np.array_equal(t, f)
        bad += not same
        epochs = nb / (n // bs)
        print(f"  n={n:>7} 배치 {bs} seed {seed} 워커 {w}: 배치 {nb} 개 ({epochs:.2f} 에포크) → "
              + ("같다" if same else f"다르다 (처음 다른 배치 {int(np.argmax((t != f).any(1)))})"))
    print("통과" if not bad else f"실패 {bad}")
    return 1 if bad else 0


# ------------------------------------------------------------------ 표 전수 대조 (GPU 없음)
_W = {}


def _table_init(tdir):
    from fasttrain import fast, orig

    _W["ref"] = orig.NonImageRef()
    _W["tab"] = fast.Table(tdir)
    _W["files"] = table_files(tdir)[0]


def _table_chunk(idx):
    ref, tab, files = _W["ref"], _W["tab"], _W["files"]
    nat = tab.samples(idx)
    bad = {k: 0 for k in FIELDS + ("req",)}
    maxd = {k: 0.0 for k in FIELDS}
    first = []
    for j, i in enumerate(idx):
        r, frames = ref(int(i))
        for k in FIELDS:
            nv = np.ones(3, bool) if k == "image_mask" else nat[k][j]
            rv = r[k]
            if nv.shape != rv.shape or nv.dtype != rv.dtype or not np.array_equal(nv, rv):
                bad[k] += 1
                if nv.shape == rv.shape:
                    maxd[k] = max(maxd[k], float(np.abs(nv.astype(np.float64) - rv.astype(np.float64)).max()))
                if len(first) < 5:
                    first.append((int(i), k))
        got = [(files[int(f)], int(fr)) for f, fr in nat["req"][j]]
        if got != frames:
            bad["req"] += 1
            if len(first) < 5:
                first.append((int(i), "req"))
    return bad, maxd, first


def cmd_table(a):
    from fasttrain import fast, orig

    cfg = orig.train_config()
    tdir = fast.ensure_table(cfg)
    n = len(fast.Table(tdir))
    idx = np.arange(n) if a.n <= 0 else np.random.default_rng(0).choice(n, a.n, replace=False)
    chunks = np.array_split(idx, max(1, len(idx) // 2000))
    t = time.time()
    ctx = multiprocessing.get_context("spawn")
    tot = {k: 0 for k in FIELDS + ("req",)}
    maxd = {k: 0.0 for k in FIELDS}
    first = []
    with ctx.Pool(a.procs, initializer=_table_init, initargs=(tdir,)) as pool:
        for k, (b, m, f) in enumerate(pool.imap_unordered(_table_chunk, chunks)):
            for x in b:
                tot[x] += b[x]
            for x in m:
                maxd[x] = max(maxd[x], m[x])
            first += f
            if k % 20 == 0:
                print(f"  ... {k + 1}/{len(chunks)} 묶음 ({time.time() - t:.0f}s)", flush=True)
    print("=" * 78)
    print(f"이미지 외 값 + 영상 프레임 번호: 네이티브 표(Rust+C++) vs 원래 파이썬, 샘플 {len(idx)} 개 ({time.time()-t:.0f}s)")
    print("=" * 78)
    for k in FIELDS + ("req",):
        v = "비트 동일" if tot[k] == 0 else "다름"
        extra = f"   다른 샘플 {tot[k]}, 최대 |오차| {maxd.get(k, 0):.3g}" if tot[k] else ""
        print(f"  {k:<26}{v}{extra}")
    if first:
        print("  처음 다른 것:", first[:5])
    bad = sum(tot.values())
    print("\n" + ("통과 — 전부 비트 동일" if not bad else f"실패 — {bad}"))
    return 1 if bad else 0


# ------------------------------------------------------------------ 구간별 (GPU)
def cmd_stage(a):
    from openpi_client import image_tools as pil_tools  # 원래 ResizeImages 가 쓰는 것 (openpi transforms.py:9, 190)
    from fasttrain import fast, lut

    root = os.path.expanduser("~/data/2026-challenge-demos/videos")
    cams = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
            "observation.rgb.right_realsense_link_camera_0"]
    videos = [f"{root}/{c}/chunk-000/file-00{k}.mp4" for c in cams for k in (0, 1)]
    eng = fast.Engine(videos, [fast.ensure_index(v) for v in videos], threads=4)
    print(f"크기 조정 커널 vs 원래 PIL, 무작위 uint8 영상 16장씩 (모양별 다른 값 수): {fast.validate_resize(eng)}")
    rng = np.random.default_rng(a.seed)
    print("=" * 78)
    print(f"구간별 대조 (파일 {len(videos)}개 × 프레임 {a.n}개, 무작위 seed={a.seed})")
    print("=" * 78)
    print(f"  {'파일':<52}{'RGB(색표)':>12}{'224²':>12}")
    bad = 0
    for f, v in enumerate(videos):
        W, H, ts, nfr = eng.info(f)
        frames = [int(x) for x in rng.integers(0, nfr, a.n)]
        frames[0], frames[-1] = 0, nfr - 1
        req = np.asarray([[f, t] for t in frames], np.int64)
        g_rgb = eng.decode(req, 1)
        g_small = eng.decode(req, 0)
        ref_rgb = lut.original_rgb(v, frames)  # 원래: torchcodec → /255 → (255*x).astype(uint8)
        ref_small = pil_tools.resize_with_pad(ref_rgb, 224, 224)  # 원래: PIL BILINEAR (CPU)
        n1, n2 = int((g_rgb != ref_rgb).sum()), int((g_small != ref_small).sum())
        bad += n1 + n2
        print(f"  {v.split('videos/')[1]:<52}{n1:>12}{n2:>12}")
    print(f"\n  (표의 숫자 = 원래와 다른 값의 개수, 0 이면 비트 동일)   엔진 {eng.stats()}")
    eng.close()
    print("통과" if bad == 0 else f"실패 — 다른 값 {bad}")
    return 1 if bad else 0


# ------------------------------------------------------------------ 기준 저장·자기 일치 (GPU: 원래 JAX 크기 조정)
def cmd_ref(a):
    from fasttrain import orig

    cfg = orig.train_config()
    ds, dc = orig.dataset(cfg)
    idx = fixed_indices(orig.lerobot_of(ds), a.n)
    os.makedirs(REF, exist_ok=True)
    t = time.time()
    cols = {}
    for i in idx:
        for k, v in flat(ds[i]).items():
            cols.setdefault(k, []).append(v)
    arrs = {k: orig.to_jax_dtype(np.stack(v)) for k, v in cols.items()}
    arrs["_idx"] = np.asarray(idx)
    path = os.path.join(REF, f"ref_{a.tag}.npz")
    np.savez(path, **arrs)
    print(f"원래 파이프라인 기준 {len(idx)}개 샘플 → {path} ({time.time()-t:.0f}s)")
    for k, v in arrs.items():
        print(f"  {k:<32}{str(v.shape):<24}{v.dtype}")
    return 0


def cmd_same(a):
    A = np.load(os.path.join(REF, f"ref_{a.a}.npz"))
    B = np.load(os.path.join(REF, f"ref_{a.b}.npz"))
    print(f"원래 파이프라인 자기 일치: {a.a} vs {a.b} (다른 프로세스)")
    bad = 0
    for k in A.files:
        v, why, _ = verdict(A[k], B[k])
        bad += v != "비트 동일"
        print(f"  {k:<32}{v:<12}{why}")
    return 1 if bad else 0


# ------------------------------------------------------------------ 네이티브 샘플 vs 기준 (GPU)
def cmd_check(a):
    from fasttrain import fast, orig

    R = np.load(os.path.join(REF, f"ref_{a.tag}.npz"))
    idx = np.asarray(R["_idx"], np.int64)
    cfg = orig.train_config()
    tdir = fast.ensure_table(cfg)
    files, idxs = table_files(tdir)
    lut_arr = None
    if a.negative == "lut":  # 음성 대조 1: 색표 대신 부동소수 BT.601 공식 (최대 3 차이 나는 흔한 구현)
        k = np.arange(1 << 24)
        y = (k >> 16).astype(np.float64) - 16
        u = ((k >> 8) & 255).astype(np.float64) - 128
        v = (k & 255).astype(np.float64) - 128
        rgb = np.stack([1.164383 * y + 1.596027 * v, 1.164383 * y - 0.391762 * u - 0.812968 * v,
                        1.164383 * y + 2.017232 * u], -1)
        lut_arr = np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(-1)
    eng = fast.Engine(files, idxs, threads=6, lut=lut_arr)
    tab = fast.Table(tdir)
    t = time.time()
    s = tab.samples(idx)
    req = s["req"].astype(np.int64)  # [N,3,2]
    if a.negative == "frame":  # 음성 대조 3: 프레임 번호를 하나 밀기 (시각 맞추기 실수)
        req[:, :, 1] = np.maximum(req[:, :, 1] - 1, 0)
    if a.negative == "resize":
        # 음성 대조 2: 크기 조정만 그럴듯한 다른 구현(JAX jax.image.resize — 처음에 기준으로 잘못 짚었던 것)으로.
        # 색표까지는 네이티브(mode 1), 크기 조정은 JAX 로 → 대조가 이 차이(±1)를 잡아야 한다.
        from openpi.shared import image_tools as jax_tools
        flat_req = req.reshape(-1, 2)
        imgs = np.stack([np.asarray(jax_tools.resize_with_pad(eng.decode(r[None], 1)[0], 224, 224))
                         for r in flat_req]).reshape(len(idx), 3, 224, 224, 3)
    else:
        imgs = eng.decode(req.reshape(-1, 2), 0).reshape(len(idx), 3, 224, 224, 3)
    F = {f"image/{nm}": imgs[:, c] for c, nm in enumerate(fast.NAMES)}
    for c, nm in enumerate(fast.NAMES):
        F[f"image_mask/{nm}"] = np.ones(len(idx), bool)
    for k in ("actions", "state", "tokenized_prompt", "tokenized_prompt_mask"):
        F[k] = s[k]
    print(f"네이티브 {len(idx)}개 샘플 {time.time()-t:.1f}s, 엔진 {eng.stats()}")
    eng.close()
    print("=" * 78)
    print(f"네이티브 vs 원래 기준 '{a.tag}' ({len(idx)} 샘플{', 음성 대조 ' + a.negative if a.negative else ''})")
    print("=" * 78)
    print(f"  {'텐서':<34}{'판정':<12}{'최대 |오차|':>12}   비고")
    bad = 0
    for k in R.files:
        if k == "_idx":
            continue
        if k not in F:
            print(f"  {k:<34}{'다름':<12}{'':>12}   네이티브에 없음")
            bad += 1
            continue
        v, why, err = verdict(F[k], R[k])
        bad += v != "비트 동일"
        print(f"  {k:<34}{v:<12}{err:>12.3g}   {why}")
    if a.negative:
        print("\n음성 대조: " + ("잡았다 (정상)" if bad else "못 잡았다 — 대조가 무디다"))
        return 0 if bad else 1
    print("\n" + ("통과 — 전부 비트 동일" if not bad else f"실패 — {bad} 개 텐서가 비트 동일이 아님"))
    return 1 if bad else 0


# ------------------------------------------------------------------ 로더 통째 (GPU)
def cmd_loader(a):
    import jax
    from fasttrain import fast, orig

    cfg = orig.train_config(num_workers=a.workers, batch_size=a.batch)
    it_o = iter(orig.loader(cfg, shuffle=True, num_batches=a.batches))
    nat = fast.NativeLoader(cfg, shuffle=True, num_batches=a.batches, batch_size=a.batch, persistent=a.workers > 0)
    it_f = iter(nat)
    bad = 0
    for b in range(a.batches):
        (oo, oa), (fo, fa) = next(it_o), next(it_f)
        lo = flat({"obs": oo.to_dict(), "actions": oa})
        lf = flat({"obs": fo.to_dict(), "actions": fa})
        for k in sorted(lo):
            if lo[k].dtype == object:
                continue
            if k not in lf:
                print(f"  배치 {b} {k:<44}네이티브에 없음")
                bad += 1
                continue
            v, why, _ = verdict(np.asarray(jax.device_get(lf[k])), np.asarray(jax.device_get(lo[k])))
            bad += v != "비트 동일"
            if b == 0 or v != "비트 동일":
                print(f"  배치 {b} {k:<44}{v:<12}{why}")
        del oo, oa, fo, fa, lo, lf
    print(f"\n네이티브 로더 통계 {nat.stats()}")
    print("통과 — 로더 출력(섞기 순서 포함) 전부 비트 동일" if not bad else f"실패 — {bad}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("index")
    sp.add_parser("order")
    s = sp.add_parser("table")
    s.add_argument("--n", type=int, default=0, help="0 = 전부")
    s.add_argument("--procs", type=int, default=16)
    s = sp.add_parser("stage")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("ref")
    s.add_argument("--tag", default="a")
    s.add_argument("--n", type=int, default=224)
    s = sp.add_parser("same")
    s.add_argument("a")
    s.add_argument("b")
    s = sp.add_parser("check")
    s.add_argument("--tag", default="a")
    s.add_argument("--negative", default="", choices=["", "lut", "resize", "frame"])
    s = sp.add_parser("loader")
    s.add_argument("--batches", type=int, default=3)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    fn = {"index": cmd_index, "order": cmd_order, "table": cmd_table, "stage": cmd_stage, "ref": cmd_ref,
          "same": cmd_same, "check": cmd_check, "loader": cmd_loader}[a.cmd]
    sys.exit(fn(a) or 0)


if __name__ == "__main__":
    main()
