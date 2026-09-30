"""Submission-server check with the official evaluator client (BEHAVIOR-1K omnigibson/eval/utils/network_utils.py
WebsocketClientPolicy, imported from the repo with omnigibson.macros stubbed):

  1. writes an observation sequence (evaluator format: RGBA camera images at the recorded native resolutions
     720x720 head / 480x480 wrists, "<robot>::proprio" [n_env, P] f32, "task_id" [n_env, 1] int64) to SEQ,
  2. drives the server with it through WebsocketClientPolicy.act / reset (n_env environments, like
     BatchedEvaluator), optionally with action chunk requests,
  3. saves the actions to OUT; tools/server_ref (the same sequence through pi05_act_batch in-process) must match
     bit for bit.

    python server_protocol_test.py --port 8000 --kind radio|pb --envs 2 --steps 60 --seq S.pi05d --out A.npy
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def official_client():
    """network_utils.WebsocketClientPolicy from the BEHAVIOR-1K checkout (only omnigibson.macros is stubbed)."""
    og = types.ModuleType("omnigibson")
    macros = types.ModuleType("omnigibson.macros")
    macros.gm = types.SimpleNamespace(DEBUG=False)
    og.macros = macros
    sys.modules.setdefault("omnigibson", og)
    sys.modules.setdefault("omnigibson.macros", macros)
    import importlib.util

    p = "/mnt/c/behavior-2026/BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py"
    spec = importlib.util.spec_from_file_location("b1k_network_utils", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.WebsocketClientPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kind", choices=["radio", "pb"], default="radio")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch as th
    from export_weights import Writer

    d = np.load("/mnt/c/behavior-2026/data/pi05_native/inputs_radio.npz")
    raw = [d["raw_head"], d["raw_left"], d["raw_right"]]  # recorded evaluator images (uint8 RGB)
    prop61 = d["proprio"]
    r = "robot_r1" if args.kind == "pb" else "robot"  # robot name of the weights (radio config: "robot")
    cams = (f"{r}::{r}:zed_link:Camera:0::rgb", f"{r}::{r}:left_realsense_link:Camera:0::rgb",
            f"{r}::{r}:right_realsense_link:Camera:0::rgb")
    E, T = args.envs, args.steps
    W = Writer()
    W.c("envs", E); W.c("steps", T); W.c("kind", args.kind)
    seq = []
    for t in range(T):
        f = [(t // 9 + 5 * e) % raw[0].shape[0] for e in range(E)]  # observations change every 9 steps
        obs = {}
        for c in range(3):
            rgba = np.concatenate([raw[c][f], np.full(raw[c][f].shape[:-1] + (1,), 255, np.uint8)], axis=-1)
            obs[cams[c]] = np.ascontiguousarray(rgba)
            W.t(f"s{t}.cam{c}", obs[cams[c]])
        obs[f"{r}::proprio"] = np.ascontiguousarray(prop61[f].astype(np.float32))
        obs["task_id"] = np.array([[0 if args.kind == "pb" else 6] for _ in range(E)], np.int64)
        W.t(f"s{t}.proprio", obs[f"{r}::proprio"])
        W.t(f"s{t}.task", obs["task_id"].astype(np.int32))
        seq.append(obs)
    W.write(pathlib.Path(args.seq), manifest_copy=False)

    Client = official_client()
    pol = Client(host=args.host, port=args.port, action_chunk_size=args.chunk)
    pol.reset()
    acts = []
    for t in range(T):
        a = pol.act({k: th.from_numpy(v) for k, v in seq[t].items()})
        acts.append(np.asarray(a, np.float32))
    np.stack(acts).astype(np.float32).tofile(args.out)  # raw f32 [steps][envs][23]
    print(f"{T} steps x {E} envs through the official WebsocketClientPolicy -> {args.out}, action shape {acts[0].shape}")


if __name__ == "__main__":
    main()
