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
