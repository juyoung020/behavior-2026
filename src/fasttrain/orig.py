"""원래 openpi 학습 데이터 파이프라인(pi05_b1k)을 이 PC 경로로 세운다.

측정(tools/ft_bench.py)과 일치 검증(tools/ft_verify.py)의 **기준**이다. openpi 코드는 한 줄도 바꾸지 않고
설정 값만 바꾼다:
  - dataset_root  : 이 PC 의 데이터 사본 (기본 ~/data/2026-challenge-demos, WSL ext4)
  - episodes      : 과제 0(turning_on_radio) 의 에피소드 0~199. 내려받은 것이 과제 0 뿐인데 meta/info.json 은
                    20000 에피소드라, 안 주면 LeRobot 이 나머지를 허브에서 받으려 한다
                    (lerobot/datasets/dataset_reader.py:152-158 → lerobot_dataset.py:260).
  - assets_base_dir: 정규화 통계. 기본 체크포인트의 assets/turning_on_radio/norm_stats.json 사본.
나머지(action_horizon 32, 변환 순서, 토크나이저, 배치 32, 워커 8, seed 42)는 pi05_b1k 그대로
(openpi src/openpi/training/config.py:757-773, 605-610).
"""
from __future__ import annotations

import dataclasses
import os

DATA_ROOT = os.environ.get("FT_DATA_ROOT", os.path.expanduser("~/data/2026-challenge-demos"))
ASSETS = os.environ.get("FT_ASSETS", os.path.expanduser("~/fasttrain_work/assets"))
TASK0_EPISODES = tuple(range(200))


def train_config(data_root: str = DATA_ROOT, assets_base: str = ASSETS, episodes=TASK0_EPISODES,
                 batch_size: int = 32, num_workers: int = 8):
    import openpi.training.config as C

    cfg = C.get_config("pi05_b1k")
    base = dataclasses.replace(
        cfg.data.base_config,
        dataset_root=data_root,
        dataset_kwargs={"tolerance_s": 5e-4, "episodes": list(episodes)},
    )
    data = dataclasses.replace(cfg.data, repo_id="turning_on_radio", base_config=base)
    return dataclasses.replace(cfg, data=data, assets_base_dir=assets_base, batch_size=batch_size,
                               num_workers=num_workers, wandb_enabled=False)


def data_config(cfg):
    return cfg.data.create(cfg.assets_dirs, cfg.model)


def transform_list(dc):
    """원래 파이프라인이 한 샘플에 적용하는 변환을 순서대로 (data_loader.py:150-153, 214-222)."""
    import openpi.transforms as T
    import openpi.training.lerobot_compat as LC

    meta = LC.LeRobotDatasetMetadata(repo_id=dc.repo_id, root=dc.dataset_root)
    return [
        T.PromptFromLeRobotTask(LC.tasks_from_metadata(meta)),
        *dc.repack_transforms.inputs,
        *dc.data_transforms.inputs,
        T.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ]


def dataset(cfg):
    """원래 파이프라인의 샘플 단위 데이터셋 (create_b1k_data_loader 의 앞 두 줄과 같다, data_loader.py:310-312)."""
    from openpi.training import data_loader as DL

    dc = data_config(cfg)
    ds = DL.create_b1k_dataset(data_config=dc, action_horizon=cfg.model.action_horizon)
    return DL.transform_dataset(ds, dc), dc


def lerobot_of(ds):
    """TransformedDataset 두 겹 안의 LeRobotDataset."""
    while hasattr(ds, "_dataset"):
        ds = ds._dataset
    return ds


def loader(cfg, *, shuffle: bool = True, num_batches: int | None = None):
    """원래 로더 그대로 (create_b1k_data_loader, data_loader.py:302-324): 워커 → collate → jax 배열 → Observation."""
    from openpi.training import data_loader as DL

    return DL.create_b1k_data_loader(cfg, shuffle=shuffle, num_batches=num_batches)


def to_jax_dtype(a):
    """원래 배치가 JAX 로 갈 때의 형 변환 (x64 꺼짐: float64→float32, int64→int32). 검증 기준을 JAX 입력과 같은 형으로."""
    import numpy as np

    a = np.asarray(a)
    if a.dtype == np.float64:
        return a.astype(np.float32)
    if a.dtype == np.int64:
        return a.astype(np.int32)
    return a


class NonImageRef:
    """원래 파이프라인의 이미지 외 값 + 영상 프레임 번호 (검증 기준, 영상 디코딩·크기 조정만 뺌).

    lerobot get_item(dataset_reader.py:292-323) 과 원래 openpi 변환 그대로. 이미지 자리에는 작은 자리표시를 넣는다 —
    이미지 외 값은 이미지에 기대지 않는다(2026-09-29 288 샘플에서 ds[i] 와 비트 동일 확인, docs/학습환경_가속.md 3.6).
    프레임 번호는 lerobot 식 round((에피소드 시작 시각 + 프레임 시각) × average_fps) (video_utils.py:305).
    """

    def __init__(self, cfg=None):
        import numpy as np
        import openpi.training.config as C
        import openpi.transforms as T
        from torchcodec.decoders import VideoDecoder

        cfg = cfg or train_config()
        ds, dc = dataset(cfg)
        self.lds = lerobot_of(ds)
        self.tfs = [t for t in transform_list(dc) if not isinstance(t, T.ResizeImages)]
        robot = C.ROBOT_REGISTRY[cfg.data.robot_config_name]
        self.cams = [robot.observations[f"image_{i}"].dataset_key for i in range(3)]
        self._fps = {}
        self._vd = VideoDecoder
        self._ph = np.zeros((3, 1, 1), np.float32)

    def fps(self, path: str) -> float:
        if path not in self._fps:  # lerobot 과 같은 방식(approximate)의 average_fps
            self._fps[path] = self._vd(path, seek_mode="approximate").metadata.average_fps
        return self._fps[path]

    def __call__(self, idx: int):
        import numpy as np

        reader, meta = self.lds.reader, self.lds.meta
        item = reader.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        q, pad = reader._get_query_indices(item["index"].item(), ep_idx)
        item = {**item, **pad, **reader._query_hf_dataset(q)}
        cur = item["timestamp"].item()
        ep = meta.episodes[ep_idx]
        frames = []
        for key in self.cams:
            path = str(self.lds.root / meta.get_video_file_path(ep_idx, key))
            frames.append((path, round((ep[f"videos/{key}/from_timestamp"] + cur) * self.fps(path))))
            item[key] = self._ph
        item["task"] = meta.tasks.iloc[item["task_index"].item()].name
        x = item
        for tf in self.tfs:
            x = tf(x)
        x.pop("image")
        out = {k: to_jax_dtype(v) for k, v in x.items() if k != "image_mask"}
        out["image_mask"] = np.asarray([bool(v) for v in x["image_mask"].values()])
        return out, frames
