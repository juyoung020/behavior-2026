"""Offline: build the reference input set from real demonstrations (task 0 = turning_on_radio).

For each sample: the three RGB frames at the same timestep (decoded with ffmpeg from the LeRobot v3
mp4 files), the 61-dim observation.state as the evaluator's `robot::proprio` stand-in, the task prompt,
and a fixed noise tensor (seeded) for the flow-matching start point.

Images are brought to 224x224 exactly the way the openpi B1K wrapper does it
(openpi_client.image_tools.resize_with_pad, PIL bilinear; eval_b1k_wrapper.py:82). The raw frames are
kept too so the native resize can be checked against PIL.

    wsl ... tools/wsl_py.sh tools/make_inputs.py --n 32 --out /mnt/c/behavior-2026/data/pi05_native/inputs_radio.npz
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
import pyarrow.parquet as pq
from openpi_client.image_tools import resize_with_pad

ROOT = pathlib.Path("/mnt/c/behavior-2026/data/2026-challenge-demos")
CAMS = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0"]  # image_0, image_1, image_2 (b1k.py robot config)


def decode_frame(path: pathlib.Path, t: float, size: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.6f}", "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8)[: size * size * 3].reshape(size, size, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    info = json.loads((ROOT / "meta/info.json").read_text())
    tasks = {json.loads(l)["task_index"]: json.loads(l) for l in (ROOT / "meta/tasks.jsonl").read_text().splitlines()}
    ep = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    idx = [i for i, ti in enumerate(ep["task_index"]) if ti == args.task_index]
    rng = np.random.default_rng(20260929)
    # spread over episodes and over the episode timeline
    picks = []
    eps = rng.choice(idx, size=args.n, replace=len(idx) < args.n)
    for k, e in enumerate(eps):
        frac = [0.05, 0.3, 0.55, 0.8][k % 4]
        picks.append((int(e), int(ep["length"][e] * frac)))

    data_tables = {}
    imgs224, raws, props, actions, meta = [], [], [], [], []
    for e, f in picks:
        dfile = ROOT / info["data_path"].format(chunk_index=ep["data/chunk_index"][e], file_index=ep["data/file_index"][e])
        if dfile not in data_tables:
            data_tables[dfile] = pq.read_table(dfile, columns=["observation.state", "action", "index"])
        tab = data_tables[dfile]
        row = ep["dataset_from_index"][e] + f
        base = tab.column("index")[0].as_py()
        props.append(np.asarray(tab.column("observation.state")[row - base].as_py(), np.float32))
        actions.append(np.asarray(tab.column("action")[row - base].as_py(), np.float32))
        views, rv = [], []
        for cam in CAMS:
            vpath = ROOT / info["video_path"].format(video_key=cam, chunk_index=ep[f"videos/{cam}/chunk_index"][e],
                                                     file_index=ep[f"videos/{cam}/file_index"][e])
            size = info["features"][cam]["shape"][0]
            t = ep[f"videos/{cam}/from_timestamp"][e] + f / info["fps"]
            fr = decode_frame(vpath, t, size)
            rv.append(fr)
            views.append(resize_with_pad(fr[None], 224, 224)[0])
        imgs224.append(np.stack(views))
        raws.append(rv)
        meta.append((e, f))
        print(f"episode {e} frame {f}")

    noise = np.stack([np.random.default_rng(1000 + i).standard_normal((32, 32)).astype(np.float32) for i in range(args.n)])
    prompt = tasks[args.task_index]["task"]
    np.savez(args.out,
             images=np.stack(imgs224),  # [N, 3, 224, 224, 3] uint8
             raw_head=np.stack([r[0] for r in raws]), raw_left=np.stack([r[1] for r in raws]),
             raw_right=np.stack([r[2] for r in raws]),
             proprio=np.stack(props),  # [N, 61] float32
             demo_action=np.stack(actions),
             noise=noise,  # [N, 32, 32] float32
             episode_frame=np.asarray(meta, np.int64),
             prompt=np.asarray(prompt))
    print(f"wrote {args.out}: prompt={prompt!r}")


if __name__ == "__main__":
    main()
