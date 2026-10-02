"""네이티브 로더 접착부 — 파이썬은 ctypes 로 C ABI 를 부르고 DLPack 으로 JAX 에 넘기는 일만 한다. torch 를 부르지 않는다.

원래 (openpi create_b1k_data_loader)                  네이티브 (libftcore.so, C++/CUDA)
---------------------------------------------------  --------------------------------------------------------------
torch DataLoader + spawn 워커 8개                      C++ 생산 스레드 1 + NVDEC 엔진 스레드 N
  샘플마다: parquet 행 읽기, 영상 6개 CPU 디코딩,        표(Rust `ftprep table`, mmap)에서 행동 창 델타·정규화·상태·토큰,
  파이썬 변환 8개, PIL 크기 조정(CPU)                    영상은 NVDEC → 색표 커널 → PIL 과 같은 정수 크기 조정 커널 (GPU 슬롯에 바로)
  np.stack → jax 배열(호스트→GPU 14.8 MB)              이미지 외 0.3 MB 만 고정 메모리 → GPU, 배치 10개 텐서를 DLPack 으로
섞기: torch randperm (seed 42)                         같은 순서를 MT19937 로 (loader.cpp Sampler)

학습 쪽에 넘기는 것은 원래와 같다: (Observation.from_dict(batch), batch["actions"]).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess

import numpy as np

WORK = os.environ.get("FT_WORK", os.path.expanduser("~/fasttrain_work"))
LIB = os.environ.get("FT_LIB", os.path.join(WORK, "build", "native", "libftcore.so"))
FTPREP = os.environ.get("FT_PREP", os.path.join(WORK, "target", "release", "ftprep"))
LUT = os.path.join(WORK, "lut", "lut_w720.bin")
# 카메라 순서 = 로봇 설정 image_0,1,2 = B1KInputs 의 이름 순서 (openpi b1k_policy.py:71-80)
NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# openpi pi05_b1k 의 샘플 변환 순서 — 이것과 다르면 표가 원래와 달라질 수 있으니 멈춘다
EXPECTED_TRANSFORMS = ("PromptFromLeRobotTask", "RepackTransform", "B1KInputs", "MappedDeltaActions", "Normalize",
                       "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt", "PadStatesAndActions")

_P = ctypes.c_void_p
_I64P = ctypes.POINTER(ctypes.c_int64)
_I32P = ctypes.POINTER(ctypes.c_int32)
_F32P = ctypes.POINTER(ctypes.c_float)
_F64P = ctypes.POINTER(ctypes.c_double)
_U8P = ctypes.POINTER(ctypes.c_uint8)
_lib = None


def lib():
    """libftcore.so 를 불러 함수 모양을 붙인다 (빌드: src/vla/fasttrain/build.sh)."""
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(LIB):
        raise FileNotFoundError(f"{LIB} 가 없다 — `bash src/vla/fasttrain/build.sh` 로 빌드")
    L = ctypes.CDLL(LIB)
    sig = {
        "ft_last_error": (ctypes.c_char_p, []),
        "ft_table_open": (_P, [ctypes.c_char_p]),
        "ft_table_close": (None, [_P]),
        "ft_table_len": (ctypes.c_int64, [_P]),
        "ft_table_dims": (ctypes.c_int, [_P, _I32P]),
        "ft_table_samples": (ctypes.c_int, [_P, _I64P, ctypes.c_int64, _F32P, _F32P, _I32P, _U8P, _I32P]),
        "ft_sampler_order": (ctypes.c_int, [ctypes.c_int64, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int64, _I64P]),
        "ft_engine_create": (_P, [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), _U8P,
                                  ctypes.c_int, ctypes.c_int]),
        "ft_engine_destroy": (None, [_P]),
        "ft_engine_info": (ctypes.c_int, [_P, ctypes.c_int, _I64P]),
        "ft_engine_stats": (ctypes.c_int, [_P, _F64P]),
        "ft_engine_decode_host": (ctypes.c_int, [_P, _I64P, ctypes.c_int, ctypes.c_int, _U8P, ctypes.c_int64]),
        "ft_engine_resize_host": (ctypes.c_int, [_P, _U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, _U8P]),
        "ft_loader_create": (_P, [ctypes.c_char_p, _U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_uint64, ctypes.c_int, ctypes.c_int]),
        "ft_loader_engine": (_P, [_P]),
        "ft_loader_next": (ctypes.c_int, [_P, _I64P, ctypes.POINTER(_P)]),
        "ft_loader_stats": (ctypes.c_int, [_P, _F64P]),
        "ft_loader_destroy": (ctypes.c_int, [_P]),
        "ft_dl_release": (None, [_P]),
    }
    for name, (res, args) in sig.items():
        f = getattr(L, name)
        f.restype, f.argtypes = res, args
    _lib = L
    return L


def _err() -> str:
    return lib().ft_last_error().decode("utf-8", "replace")


def _ptr(a: np.ndarray, t):
    assert a.flags["C_CONTIGUOUS"]
    return a.ctypes.data_as(t)


# ---------------------------------------------------------------- 표 (Rust ftprep table)
def table_spec(cfg) -> tuple[dict, bytes]:
    """openpi 설정에서 표 스펙을 뽑는다 (학습 시작 때 한 번. openpi 설정 모듈을 읽으므로 여기만 openpi 를 import)."""
    import openpi.models.model as _model
    import openpi.training.config as C
    import openpi.training.lerobot_compat as LC
    from openpi.shared import download

    from fasttrain import orig

    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = tuple(type(t).__name__ for t in orig.transform_list(dc))
    if names != EXPECTED_TRANSFORMS:
        raise RuntimeError(f"openpi 변환 순서가 표가 가정한 것과 다르다: {names}")
    if cfg.model.model_type != _model.ModelType.PI05 or dc.use_quantile_norm or not dc.prompt_from_task:
        raise RuntimeError("pi05 + z-score 정규화 + 과제 문장 설정만 지원")
    robot = C.ROBOT_REGISTRY[cfg.data.robot_config_name]
    cams = [robot.observations[f"image_{i}"].dataset_key for i in range(3)]
    delta = cfg.data._build_delta_mappings(robot) if cfg.data.extra_delta_transform else []
    S = sum(1 if p.is_eef else len(p.indices) for p in robot.proprio)
    A = robot.action_dim
    ns = dc.norm_stats
    norm = np.concatenate([np.asarray(ns["state"].mean, np.float64)[:S], np.asarray(ns["state"].std, np.float64)[:S],
                           np.asarray(ns["actions"].mean, np.float64)[:A], np.asarray(ns["actions"].std, np.float64)[:A]])
    meta = LC.LeRobotDatasetMetadata(repo_id=dc.repo_id, root=dc.dataset_root)
    prompts = {str(k): v for k, v in LC.tasks_from_metadata(meta).items()}
    tok = str(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
    kw = dc.dataset_kwargs or {}
    spec = {
        "root": str(dc.dataset_root).rstrip("/"),
        "episodes": [int(e) for e in kw["episodes"]] if kw.get("episodes") is not None else None,
        "cams": cams,
        "state_key": "observation.state",
        "action_key": robot.action_key,
        "proprio": [{"indices": list(map(int, p.indices)), "eef": bool(p.is_eef)} for p in robot.proprio],
        "delta": [{"action": list(map(int, a)), "state": list(map(int, s))} for a, s in delta],
        "action_dim": int(A),
        "model_dim": int(cfg.model.action_dim),
        "horizon": int(cfg.model.action_horizon),
        "prompts": prompts,
        "tokenizer": tok,
        "max_token_len": int(cfg.model.max_token_len),
        "discrete_state": bool(cfg.model.discrete_state_input),
        "tolerance_s": float(kw.get("tolerance_s", 1e-4)),
    }
    return spec, norm.astype("<f8").tobytes()


def ensure_table(cfg, force: bool = False) -> str:
    """표가 없거나 스펙이 바뀌었으면 Rust ftprep 로 만든다. 표 폴더 경로를 돌려준다."""
    spec, norm = table_spec(cfg)
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode() + norm).hexdigest()[:12]
    out = os.path.join(WORK, "cache", f"{cfg.name}-{cfg.data.repo_id}-{h}")
    if os.path.exists(os.path.join(out, "table.txt")) and not force:
        return out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "norm.f64"), "wb") as f:
        f.write(norm)
    spec.update(out=out, idx_dir=os.path.join(WORK, "idx"), norm_file=os.path.join(out, "norm.f64"))
    with open(os.path.join(out, "spec.json"), "w") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    subprocess.run([FTPREP, "table", os.path.join(out, "spec.json")], check=True)
    return out


class Table:
    """표 읽기 (CPU, 검증용): 샘플 번호 → 이미지 외 값 (네이티브 로더가 GPU 에 넣는 것과 같은 함수)."""

    def __init__(self, path: str):
        self.h = lib().ft_table_open(path.encode())
        if not self.h:
            raise RuntimeError(_err())
        d = np.zeros(6, np.int32)
        lib().ft_table_dims(self.h, _ptr(d, _I32P))
        self.S, self.A, self.M, self.H, self.T, self.cams = map(int, d)
        self.n = int(lib().ft_table_len(self.h))

    def __len__(self):
        return self.n

    def samples(self, idx) -> dict:
        idx = np.ascontiguousarray(idx, np.int64)
        k = len(idx)
        act = np.zeros((k, self.H, self.M), np.float32)
        st = np.zeros((k, self.M), np.float32)
        tok = np.zeros((k, self.T), np.int32)
        tm = np.zeros((k, self.T), np.uint8)
        req = np.zeros((k, self.cams, 2), np.int32)
        if lib().ft_table_samples(self.h, _ptr(idx, _I64P), k, _ptr(act, _F32P), _ptr(st, _F32P), _ptr(tok, _I32P),
                                  _ptr(tm, _U8P), _ptr(req, _I32P)):
            raise RuntimeError(_err())
        return {"actions": act, "state": st, "tokenized_prompt": tok, "tokenized_prompt_mask": tm.astype(bool),
                "req": req}

    def close(self):
        if self.h:
            lib().ft_table_close(self.h)
            self.h = None


def sampler_order(n: int, bs: int, seed: int, nbatches: int, shuffle: bool = True, persistent: bool = True):
    out = np.zeros((nbatches, bs), np.int64)
    if lib().ft_sampler_order(n, bs, seed, int(shuffle), int(persistent), nbatches, _ptr(out, _I64P)):
        raise RuntimeError(_err())
    return out


# ---------------------------------------------------------------- 엔진 (검증용 직접 사용)
def load_lut() -> np.ndarray:
    if not os.path.exists(LUT):
        raise FileNotFoundError(f"{LUT} 가 없다 — `ft_run.sh src/vla/fasttrain/lut.py build`")
    a = np.fromfile(LUT, np.uint8)
    assert a.size == (1 << 24) * 3
    return a


def validate_resize(engine, n: int = 16, seed: int = 1) -> dict:
    """크기 조정 커널 == 원래 (openpi_client.image_tools.resize_with_pad, PIL). 무작위 uint8 영상, 크기별 다른 값 수 (0 이어야)."""
    from openpi_client import image_tools as pil_tools

    rng = np.random.default_rng(seed)
    res = {}
    for h, w in ((720, 720), (480, 480), (480, 640), (300, 200)):  # 정사각 두 가지 + 붙이기(패딩) 가 생기는 모양
        X = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
        ref = pil_tools.resize_with_pad(X, 224, 224)
        res[f"{h}x{w}"] = int((engine.resize(X) != ref).sum())
    return res


class Engine:
    """NVDEC 엔진을 파일 목록으로 직접 (구간 대조용)."""

    def __init__(self, videos, indexes, threads: int = 4, device: int = 0, lut=None):
        lut = load_lut() if lut is None else np.ascontiguousarray(lut, np.uint8)
        v = (ctypes.c_char_p * len(videos))(*[p.encode() for p in videos])
        ix = (ctypes.c_char_p * len(indexes))(*[p.encode() for p in indexes])
        self.h = lib().ft_engine_create(len(videos), v, ix, _ptr(lut, _U8P), threads, device)
        if not self.h:
            raise RuntimeError(_err())
        self.owned = True

    @classmethod
    def borrow(cls, handle):
        e = cls.__new__(cls)
        e.h, e.owned = handle, False
        return e

    def info(self, f: int):
        o = np.zeros(4, np.int64)
        if lib().ft_engine_info(self.h, f, _ptr(o, _I64P)):
            raise RuntimeError(_err())
        return [int(x) for x in o]  # W, H, timescale, 표본 수

    def stats(self):
        o = np.zeros(4, np.float64)
        lib().ft_engine_stats(self.h, _ptr(o, _F64P))
        return {"requests": o[0], "decoded_frames": o[1], "parsers": o[2], "map_s": o[3]}

    def decode(self, req, mode: int = 0) -> np.ndarray:
        req = np.ascontiguousarray(req, np.int64).reshape(-1, 2)
        if mode == 0:
            out = np.empty((len(req), 224, 224, 3), np.uint8)
        else:
            W, H, _, _ = self.info(int(req[0, 0]))
            out = np.empty((len(req), H, W, 3), np.uint8)
        if lib().ft_engine_decode_host(self.h, _ptr(req, _I64P), len(req), mode, _ptr(out, _U8P), out.nbytes):
            raise RuntimeError(_err())
        return out

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 [n,H,W,3] → [n,224,224,3] (GPU 크기 조정 커널, 원래 PIL 과 같아야 한다)."""
        rgb = np.ascontiguousarray(rgb, np.uint8)
        out = np.empty((len(rgb), 224, 224, 3), np.uint8)
        if lib().ft_engine_resize_host(self.h, _ptr(rgb, _U8P), len(rgb), rgb.shape[2], rgb.shape[1], _ptr(out, _U8P)):
            raise RuntimeError(_err())
        return out

    def close(self):
        if self.h and self.owned:
            lib().ft_engine_destroy(self.h)
        self.h = None


def index_path(video: str) -> str:
    """Rust ftprep 의 index_path 와 같은 이름 (videos/ 아래 상대 경로의 / → .)."""
    rel = video.split("/videos/", 1)[1]
    return os.path.join(WORK, "idx", rel.replace("/", ".") + ".ftidx")


def ensure_index(video: str) -> str:
    out = index_path(video)
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(video):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([FTPREP, "index", video, out], check=True, stdout=subprocess.DEVNULL)
    return out


# ---------------------------------------------------------------- DLPack → JAX
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)


class _DL:
    """C 가 만든 DLManagedTensor 하나를 __dlpack__ 규약으로 감싼다. JAX 가 가져가면(복사 없음) 다 쓴 뒤 deleter 를 부른다."""

    __slots__ = ("ptr", "dev")

    def __init__(self, ptr: int, dev: tuple[int, int]):
        self.ptr, self.dev = ptr, dev

    def __dlpack__(self, stream=None, **_):  # 데이터는 C 쪽에서 이미 동기화돼 있다
        return _capsule_new(self.ptr, b"dltensor", None)

    def __dlpack_device__(self):
        return self.dev


# ---------------------------------------------------------------- 로더
class NativeLoader:
    """openpi TorchDataLoader + DataLoaderImpl 자리에 들어간다: (Observation, actions) 를 낸다."""

    def __init__(self, cfg, *, sharding=None, shuffle=False, num_batches=None, batch_size=None, decode_threads=4,
                 slots=6, persistent=True, device=0, check=True):
        import jax

        self.cfg = cfg
        self.data_config_ = cfg.data.create(cfg.assets_dirs, cfg.model)
        self.table_dir = ensure_table(cfg)
        self.bs = (batch_size or cfg.batch_size) // jax.process_count()
        self.num_batches = num_batches
        self.device = device
        lut = load_lut()
        # persistent=True: 원래 워커 > 0 (openpi 기본 8) 일 때의 섞기 순서
        self.h = lib().ft_loader_create(self.table_dir.encode(), _ptr(lut, _U8P), decode_threads, device, self.bs,
                                        int(shuffle), cfg.seed, int(persistent), slots)
        if not self.h:
            raise RuntimeError(_err())
        self.engine = Engine.borrow(lib().ft_loader_engine(self.h))
        if check:  # 시작 때 한 번: 크기 조정 커널 == 원래 PIL (다르면 멈춘다)
            bad = validate_resize(self.engine)
            if any(bad.values()):
                raise RuntimeError(f"크기 조정 커널이 원래(PIL) 결과와 다르다 {bad}")
        if sharding is None:
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)),
                                                  jax.sharding.PartitionSpec("B"))
        self.sharding = sharding
        self.last_indices = None

    def data_config(self):
        return self.data_config_

    def stats(self) -> dict:
        o = np.zeros(4, np.float64)
        lib().ft_loader_stats(self.h, _ptr(o, _F64P))
        return {"batches": o[0], "consumer_wait_s": o[1], "producer_wait_slot_s": o[2], "fill_s": o[3],
                **self.engine.stats()}

    def next_raw(self):
        """다음 배치를 JAX 배열 dict 로 (원래 collate 뒤 + GPU 전송 뒤와 같은 모양·형)."""
        import jax

        idx = np.empty(self.bs, np.int64)
        ptrs = (_P * 10)()
        if lib().ft_loader_next(self.h, _ptr(idx, _I64P), ptrs):
            raise RuntimeError(_err())
        dev = (2, self.device)  # kDLCUDA
        arrs = []
        try:
            for k in range(10):
                arrs.append(jax.dlpack.from_dlpack(_DL(ptrs[k], dev)))
        except BaseException:
            for k in range(len(arrs), 10):
                lib().ft_dl_release(ptrs[k])
            raise
        batch = {
            "image": dict(zip(NAMES, arrs[0:3])),
            "image_mask": dict(zip(NAMES, arrs[3:6])),
            "state": arrs[6],
            "tokenized_prompt": arrs[7],
            "tokenized_prompt_mask": arrs[8],
            "actions": arrs[9],
        }
        self.last_indices = idx
        return jax.tree.map(lambda a: jax.device_put(a, self.sharding), batch)

    def __iter__(self):
        import openpi.models.model as _model

        n = 0
        while self.num_batches is None or n < self.num_batches:
            b = self.next_raw()
            n += 1
            yield _model.Observation.from_dict(b), b["actions"]

    def close(self):
        """JAX 가 아직 배치를 들고 있으면 지우지 않는다 (프로세스 끝까지 남김)."""
        if self.h and lib().ft_loader_destroy(self.h) == 0:
            self.h = None


def loader(batch_size=32, shuffle=True, decode_threads=4, num_batches=None, slots=6, cfg=None):
    from fasttrain import orig

    cfg = cfg or orig.train_config(batch_size=batch_size)
    return NativeLoader(cfg, shuffle=shuffle, num_batches=num_batches, batch_size=batch_size,
                        decode_threads=decode_threads, slots=slots)


def create_fast_b1k_data_loader(config, *, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False):
    """openpi data_loader.create_b1k_data_loader 와 같은 모양 (train_b1k.py 에서 FT_FAST_DATA=1 일 때)."""
    assert not skip_norm_stats, "skip_norm_stats 는 지원 안 함"
    # 디코딩 스레드 4 가 이 PC(RTX 5070 Ti)에서 가장 빨랐다 (2: 185, 4: 289, 6: 264, 8: 249 샘플/s — docs/학습환경_가속.md 4.1)
    threads = int(os.environ.get("FT_DECODE_THREADS", "4"))
    slots = int(os.environ.get("FT_SLOTS", "6"))
    # 원래 섞기 순서는 워커 수에 따라 다르다: 워커 > 0 이면 지속 반복자(기준 시드 한 번), 0 이면 에포크마다
    return NativeLoader(config, sharding=sharding, shuffle=shuffle, num_batches=num_batches, decode_threads=threads,
                        slots=slots, persistent=config.num_workers > 0)
