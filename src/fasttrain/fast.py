"""가속 로더 — 파이썬은 openpi 로더에 거는 접착부만 한다.

원래 (openpi create_b1k_data_loader)                     가속
------------------------------------------------------  ----------------------------------------------------------
워커: 행 읽기 + 영상 6개 디코딩(깊이 3개는 버림)          워커: 행 읽기 + 영상 대신 "어느 파일의 몇 번 프레임" 계산
      + 변환(JAX 크기 조정 포함, 워커마다 GPU 문맥)             + 원래 변환 그대로(크기 조정만 뺌, GPU 안 씀)
메인: np.stack → jax 배열(호스트→GPU)                     메인(뒤 스레드): C++ 엔진이 NVDEC 로 3카메라×배치 프레임을
                                                               풀고 색 변환·크기 조정 커널로 GPU 에 바로 224² 를 쓴다
                                                               → DLPack 으로 JAX 에 넘김. 다음 배치를 미리 만든다.
이미지 외 값(상태·행동·토큰)은 원래 openpi 코드가 그대로 만든다 → 비트 동일이 구조상 보장된다.
이미지는 tools/ft_verify.py 가 원래 결과와 비트 단위로 대조한다.
"""
from __future__ import annotations

import os
import queue
import threading

import numpy as np

WORK = os.environ.get("FT_WORK", os.path.expanduser("~/fasttrain_work"))
NVHDR = os.environ.get("FT_NVHDR", os.path.join(WORK, "third_party", "nv-codec-headers", "include"))
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SIZES = (720, 480)
_placeholder = np.zeros((3, 1, 1), np.float32)  # B1KInputs 가 받는 자리표시 (이미지는 GPU 에서 채운다)


# ---------------------------------------------------------------- 빌드·준비
def build_ext(verbose: bool = False):
    """C++/CUDA 확장을 빌드(처음 한 번)하고 불러온다. 외부 GPU: FT_CUDA_ARCH 로 아키텍처 지정 (예 "9.0")."""
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("FT_CUDA_ARCH", os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0"))
    from torch.utils.cpp_extension import load

    bdir = os.path.join(WORK, "build", "ftcore")
    os.makedirs(bdir, exist_ok=True)
    return load(name="ftcore", sources=[f"{CSRC}/ftcore.cpp", f"{CSRC}/nvdec.cpp", f"{CSRC}/kernels.cu"],
                extra_include_paths=[CSRC, NVHDR], extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                extra_ldflags=["-ldl"], build_directory=bdir, verbose=verbose)


def make_engine(videos, indexes, threads: int, lut_override=None, weight_hook=None, check: bool = True):
    """C++ 엔진 만들기: 색 변환 표 + 이 GPU 에서 보정한 크기 조정 탭(calib.py) → 원래와 비트 동일한지 uint8 로 점검."""
    import torch

    from fasttrain import calib

    ext = build_ext()
    lut = lut_override if lut_override is not None else torch.from_file(
        os.path.join(WORK, "lut", "lut_w720.bin"), size=(1 << 24) * 3, dtype=torch.uint8)
    lut = lut.cuda()
    st, ct, wt, sr, sc = calib.calibrate_all(SIZES)
    if weight_hook is not None:
        wt = weight_hook(wt)
    t = torch.from_numpy
    eng = ext.Engine(videos, indexes, lut, list(SIZES), t(st), t(ct), t(wt), t(sr), t(sc), threads=threads, device=0)
    eng._keep = lut  # 표가 엔진보다 먼저 풀리지 않게
    eng.calib = {m: (calib.boundaries(st[c], sr[c]), calib.boundaries(st[c], sc[c])) for c, m in enumerate(SIZES)}
    if check:
        bad = calib.validate(eng)
        if any(bad.values()):
            raise RuntimeError(f"크기 조정 커널이 원래 JAX 결과와 다르다 (다른 값 수 {bad}) — 보정 실패")
        eng.validated = bad
    return eng


def ensure_index(video_path: str) -> str:
    """Rust ftprep 로 mp4 패킷 색인을 만든다 (없을 때만)."""
    import subprocess

    rel = os.path.relpath(video_path, os.path.dirname(os.path.dirname(os.path.dirname(video_path))))
    out = os.path.join(WORK, "idx", rel.replace("/", ".") + ".ftidx")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(video_path):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([os.path.join(WORK, "target", "release", "ftprep"), "index", video_path, out], check=True,
                       stdout=subprocess.DEVNULL)
    return out


# ---------------------------------------------------------------- 워커 쪽 데이터셋
class FastDataset:
    """원래 샘플에서 영상 디코딩과 크기 조정만 뺀 것. 이미지 자리에는 (파일, 프레임) 요청을 담는다."""

    def __init__(self, cfg):
        import openpi.training.config as C
        import openpi.transforms as T
        from torchcodec.decoders import VideoDecoder

        from fasttrain import orig

        ds, dc = orig.dataset(cfg)
        self.lds = orig.lerobot_of(ds)
        self.tfs = [t for t in orig.transform_list(dc) if not isinstance(t, T.ResizeImages)]
        robot = C.ROBOT_REGISTRY[cfg.data.robot_config_name]
        self.cams = [robot.observations[f"image_{i}"].dataset_key for i in range(3)]
        meta = self.lds.meta
        self.tol = self.lds.tolerance_s
        root = str(self.lds.root)
        # 파일 표: 카메라 × (chunk, file) → 번호
        self.videos, fid = [], {}
        eps = sorted(self.lds.episodes) if self.lds.episodes is not None else range(meta.total_episodes)
        self.ep_file, self.ep_from = {}, {}
        for e in eps:
            ep = meta.episodes[e]
            fs, frs = [], []
            for key in self.cams:
                path = os.path.join(root, meta.get_video_file_path(e, key))
                if path not in fid:
                    fid[path] = len(self.videos)
                    self.videos.append(path)
                fs.append(fid[path])
                frs.append(ep[f"videos/{key}/from_timestamp"])
            self.ep_file[e], self.ep_from[e] = fs, frs
        # lerobot 이 쓰는 값 그대로: torchcodec(approximate) 의 average_fps (video_utils.py:299-302)
        self.fps = [VideoDecoder(p, seek_mode="approximate").metadata.average_fps for p in self.videos]
        self.indexes = [ensure_index(p) for p in self.videos]
        self.pts_s = None  # 엔진에서 받아 채운다 (set_pts)

    def set_pts(self, pts_seconds):
        self.pts_s = pts_seconds

    def __len__(self):
        return len(self.lds)

    def __getitem__(self, idx):
        reader, meta = self.lds.reader, self.lds.meta
        # --- lerobot dataset_reader.get_item 과 같은 순서 (292-306), 영상 디코딩만 뺌
        item = reader.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()
        q, pad = reader._get_query_indices(abs_idx, ep_idx)
        qr = reader._query_hf_dataset(q)
        item = {**item, **pad}
        for k, v in qr.items():
            item[k] = v
        cur = item["timestamp"].item()
        req = np.zeros((3, 2), np.int64)
        for c, key in enumerate(self.cams):
            f = self.ep_file[ep_idx][c]
            ts = self.ep_from[ep_idx][c] + cur
            fr = round(ts * self.fps[f])  # video_utils.py:302 와 같은 식
            loaded = self.pts_s[f][fr]
            if not abs(ts - loaded) < self.tol:  # video_utils.py:314-318 와 같은 검사
                raise ValueError(f"프레임 시각 허용오차 초과: {ts} vs {loaded} ({self.videos[f]})")
            req[c] = (f, fr)
            item[key] = _placeholder
        item["task"] = meta.tasks.iloc[item["task_index"].item()].name
        x = item
        for tf in self.tfs:
            x = tf(x)
        x["_ft_req"] = req
        return x


# ---------------------------------------------------------------- 메인 쪽 로더
class FastLoader:
    """openpi TorchDataLoader + DataLoaderImpl 자리에 그대로 들어간다: (Observation, actions) 를 낸다."""

    def __init__(self, cfg, *, sharding=None, shuffle=False, num_batches=None, num_workers=None, batch_size=None,
                 decode_threads=6, prefetch=2):
        import jax
        import torch

        from fasttrain import orig
        from openpi.training import data_loader as DL

        self.cfg = cfg
        self.data_config_ = orig.data_config(cfg)
        self.ds = FastDataset(cfg)
        self.engine = make_engine(self.ds.videos, self.ds.indexes, decode_threads)
        self.ds.set_pts([np.asarray(self.engine.pts(i), np.int64) / self.engine.info(i)[2]
                         for i in range(len(self.ds.videos))])
        bs = (batch_size or cfg.batch_size) // jax.process_count()
        nw = cfg.num_workers if num_workers is None else num_workers
        self.bs, self.num_batches, self.prefetch = bs, num_batches, prefetch
        if sharding is None:
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)),
                                                  jax.sharding.PartitionSpec("B"))
        self.sharding = sharding
        g = torch.Generator()
        g.manual_seed(cfg.seed)  # 원래와 같은 순서 (data_loader.py:488-489)
        import multiprocessing

        self.torch_loader = torch.utils.data.DataLoader(
            self.ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
            multiprocessing_context=multiprocessing.get_context("spawn") if nw > 0 else None,
            persistent_workers=nw > 0, collate_fn=DL._collate_fn, worker_init_fn=DL._worker_init_fn,
            drop_last=True, generator=g)

    def data_config(self):
        return self.data_config_

    def _cpu_batches(self):
        # 원래 TorchDataLoader.__iter__ 처럼 끝나면 처음부터 다시 돈다 (data_loader.py:508-519)
        n = 0
        while True:
            for b in self.torch_loader:
                if self.num_batches is not None and n >= self.num_batches:
                    return
                n += 1
                yield b

    def _to_device(self, b):
        import jax
        import torch

        req = torch.from_numpy(np.ascontiguousarray(b.pop("_ft_req").transpose(1, 0, 2).reshape(-1, 2)))
        B = self.bs
        out = torch.empty((3, B, 224, 224, 3), dtype=torch.uint8, device="cuda")
        self.engine.run(req, out, 0)  # GIL 을 풀고 NVDEC·커널 — 끝나면 out 이 채워져 있다
        imgs = b["image"]
        for c, name in enumerate(imgs.keys()):
            imgs[name] = jax.device_put(jax.dlpack.from_dlpack(out[c]), self.sharding)
        return jax.tree.map(
            lambda x: x if isinstance(x, jax.Array) else jax.make_array_from_process_local_data(self.sharding, x), b)

    def __iter__(self):
        import openpi.models.model as _model

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        END = object()

        def producer():
            try:
                for b in self._cpu_batches():
                    if stop.is_set():
                        return
                    q.put(self._to_device(b))
                q.put(END)
            except BaseException as e:  # 소비 쪽에서 다시 던진다
                q.put(e)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield _model.Observation.from_dict(item), item["actions"]
        finally:
            stop.set()


def loader(num_workers=2, batch_size=32, shuffle=True, decode_threads=6, num_batches=None, cfg=None):
    from fasttrain import orig

    cfg = cfg or orig.train_config(batch_size=batch_size, num_workers=num_workers)
    return FastLoader(cfg, shuffle=shuffle, num_batches=num_batches, num_workers=num_workers, batch_size=batch_size,
                      decode_threads=decode_threads)


def create_fast_b1k_data_loader(config, *, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False,
                                decode_threads=None):
    """openpi data_loader.create_b1k_data_loader 와 같은 모양 (fast-data 브랜치의 train_b1k.py 가 부른다)."""
    assert not skip_norm_stats, "skip_norm_stats 는 지원 안 함"
    threads = decode_threads or int(os.environ.get("FT_DECODE_THREADS", "6"))
    return FastLoader(config, sharding=sharding, shuffle=shuffle, num_batches=num_batches, decode_threads=threads)
