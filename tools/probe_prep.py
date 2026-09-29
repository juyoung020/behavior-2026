"""지시 형식 오프라인 실험 — 시연 데이터 준비 (일회성 도구, GPU 안 씀).

radio(과제 0) 시연에서 단계(skill) 구간마다 몇 프레임을 골라, π0.5 가 평가 때 받는 것과 같은 입력
(카메라 3장을 openpi_client resize_with_pad 로 224², proprio 61)과 정답 행동 묶음(그 프레임부터 H 스텝)을 뽑는다.
단계 정보(기술·물체·구간)와 그 프레임부터 구간 끝까지의 이동량(base_qvel 적분, 로봇 기준)도 같이 적는다.

출력: <out>.npz (head/left/right uint8 [N,224,224,3], state f32 [N,61], gt f32 [N,H,23]) + <out>.json (표본 목록)
  ~/openpi-comet/.venv/bin/python /mnt/c/behavior-2026/tools/probe_prep.py --episodes 20 --per_stage 2 --out /mnt/c/behavior-2026/data/probe/radio_e20_p2
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from openpi_client.image_tools import resize_with_pad

ROOT = "/mnt/c/behavior-2026/data/2026-challenge-demos"
CAMS = {
    "head": "observation.rgb.zed_link_camera_0",
    "left": "observation.rgb.left_realsense_link_camera_0",
    "right": "observation.rgb.right_realsense_link_camera_0",
}
FPS = 30


def integrate_base(qvel):
    """로봇 기준 속도(vx, vy, wz)를 적분해 시작 자세 기준 이동량 (dx, dy, dyaw) 을 낸다 (앞 +x, 왼쪽 +y, 반시계 +yaw)."""
    x = y = th = 0.0
    dt = 1.0 / FPS
    for vx, vy, wz in qvel:
        c, s = math.cos(th), math.sin(th)
        x += (c * vx - s * vy) * dt
        y += (s * vx + c * vy) * dt
        th += wz * dt
    return x, y, th


def read_frames(path, t_sec_list):
    cap = cv2.VideoCapture(path)
    out = []
    for t in t_sec_list:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"{path} @ {t}s 읽기 실패")
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--per_stage", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    eps = pq.read_table(f"{ROOT}/meta/episodes/chunk-000/file-000.parquet").to_pylist()
    eps = [e for e in eps if e["task_index"] == 0][: args.episodes]
    tables = {}
    samples, heads, lefts, rights, states, gts = [], [], [], [], [], []
    for e in eps:
        dfile = f"{ROOT}/data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet"
        if dfile not in tables:
            t = pq.read_table(dfile, columns=["index", "observation.state", "action"])
            tables[dfile] = (
                t.column("index").to_numpy(),
                np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32),
                np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32),
            )
        idx, st, ac = tables[dfile]
        base = int(np.searchsorted(idx, e["dataset_from_index"]))
        length = e["length"]
        ep_state, ep_act = st[base : base + length], ac[base : base + length]
        ann = json.load(open(f"{ROOT}/{e['annotation_path']}"))
        stages = ann["skill_annotation"]
        for si, sk in enumerate(stages):
            f0, f1 = sk["frame_duration"]
            for k in range(args.per_stage):
                fr = int(f0 + (f1 - f0) * (k + 1) / (args.per_stage + 1))
                if fr + args.horizon > length:
                    continue
                dx, dy, dyaw = integrate_base(ep_state[fr:f1, 0:3])
                samples.append(
                    {
                        "id": len(samples),
                        "episode_index": e["episode_index"],
                        "frame": fr,
                        "stage_idx": si,
                        "skill": sk["skill_description"][0],
                        "objects": sk["object_id"][0],
                        "stage_frames": [f0, f1],
                        "remaining_base": {"dx_m": dx, "dy_m": dy, "dyaw_deg": math.degrees(dyaw)},
                        "all_skills": [s["skill_description"][0] for s in stages],
                        "all_objects": [s["object_id"][0] for s in stages],
                    }
                )
                states.append(ep_state[fr])
                gts.append(ep_act[fr : fr + args.horizon])
        # 영상: 이 에피소드에서 고른 프레임만
        mine = [s for s in samples if s["episode_index"] == e["episode_index"]]
        for cam, key in CAMS.items():
            vpath = f"{ROOT}/videos/{key}/chunk-{e[f'videos/{key}/chunk_index']:03d}/file-{e[f'videos/{key}/file_index']:03d}.mp4"
            t0 = e[f"videos/{key}/from_timestamp"]
            frames = read_frames(vpath, [t0 + s["frame"] / FPS for s in mine])
            small = [resize_with_pad(f[None], 224, 224)[0] for f in frames]
            {"head": heads, "left": lefts, "right": rights}[cam].extend(small)
        print(f"episode {e['episode_index']}: 단계 {len(stages)}, 표본 누계 {len(samples)}", flush=True)

    np.savez(
        args.out + ".npz",
        head=np.stack(heads),
        left=np.stack(lefts),
        right=np.stack(rights),
        state=np.stack(states),
        gt=np.stack(gts),
    )
    json.dump({"horizon": args.horizon, "samples": samples}, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)
    print(f"표본 {len(samples)}개 -> {args.out}.npz/.json")


if __name__ == "__main__":
    main()
