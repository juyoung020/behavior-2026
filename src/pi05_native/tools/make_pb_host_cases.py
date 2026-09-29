"""Offline, CPU only: reference cases for the PiBehavior host side (tools/pb_host_test.cpp), produced by the 1st-place
Python code itself (alstar8 fork): state extraction, state normalization + tokens, initial-action transform,
output transform, cubic compression (scipy interp1d), correction rules, gripper-variation check, stage voting.

    wsl_py.sh make_pb_host_cases.py --out /mnt/c/behavior-2026/data/pi05_native/pb_host_cases.pi05d
"""
import argparse
import json
import pathlib
import sys
from collections import deque

sys.path.insert(0, "/mnt/c/behavior-2026/refs/behavior-1k-solution_alstar8/src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from export_weights import Writer  # noqa: E402
from openpi import transforms as T  # noqa: E402
from b1k.shared.b1k_proprio import extract_state_from_proprio, legacy_proprio_to_compact  # noqa: E402
from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation  # noqa: E402
from b1k.shared.normalize import NormStats  # noqa: E402
from b1k.transforms_normalize import NormalizeWithPerTimestamp, UnnormalizeWithPerTimestamp  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402

STAGES_2025 = (5, 6, 15, 15, 14, 12, 9, 15, 10, 15, 7, 13, 10, 15, 15, 15, 15, 11, 13, 12, 14, 15, 9, 15, 15, 15,
               15, 15, 15, 15, 11, 10, 10, 13, 5, 5, 14, 6, 8, 10, 5, 15, 8, 15, 12, 11, 9, 14, 15, 15)


def load_stats(path):
    ns = json.loads(pathlib.Path(path).read_text())
    ns = ns.get("norm_stats", ns)
    out = {}
    for k in ("state", "actions"):
        d = {kk: (np.asarray(v) if v is not None else None) for kk, v in ns[k].items()}
        out[k] = NormStats(**{kk: d.get(kk) for kk in ("mean", "std", "q01", "q99", "per_timestamp_mean",
                                                        "per_timestamp_std", "per_timestamp_q01", "per_timestamp_q99")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path.home() / "checkpoints/behavior_submission/checkpoint_2"))
    ap.add_argument("--inputs", default="/mnt/c/behavior-2026/data/pi05_native/inputs_radio.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stats = load_stats(pathlib.Path(args.ckpt) / "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json")
    norm = NormalizeWithPerTimestamp(stats, use_quantiles=False, use_per_timestamp=True)
    unnorm = UnnormalizeWithPerTimestamp(stats, use_quantiles=False, use_per_timestamp=True)
    mask = T.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
    delta, absolute = T.DeltaActions(mask), T.AbsoluteActions(mask)
    pad = T.PadStatesAndActions(32)
    rng = np.random.default_rng(7)
    data = np.load(args.inputs)
    W = Writer()
    props = data["proprio"].astype(np.float32)
    # legacy 256-d cases too: random vectors in the legacy layout
    legacy = rng.normal(size=(8, 256)).astype(np.float32)
    W.t("prop61", props)
    W.t("prop256", legacy)
    st61 = np.stack([extract_state_from_proprio(legacy_proprio_to_compact(p)) for p in props]).astype(np.float32)
    st256 = np.stack([extract_state_from_proprio(legacy_proprio_to_compact(p)) for p in legacy]).astype(np.float32)
    W.t("state61", st61)
    W.t("state256", st256)
    # normalization + tokens (pi_behavior.py:594-595 in f32)
    ns, toks = [], []
    for s in st61:
        d = pad(norm({"state": s.copy()}))
        x = np.asarray(d["state"], np.float32)
        ns.append(x)
        t = np.clip(np.digitize(x, np.linspace(-1, 1, 257, dtype=np.float32)[:-1]) - 1, 0, 255)
        toks.append(t.astype(np.int32))
    W.t("state_norm", np.stack(ns))
    W.t("state_tokens", np.stack(toks))
    # initial actions (keep 4) and output transform, with random model outputs
    kept = rng.normal(size=(len(st61), 4, 23)) * 0.3
    init, outs, raws = [], [], []
    for i, s in enumerate(st61):
        d = {"state": s.copy(), "actions": kept[i].copy()}
        d = pad(norm(delta(d)))
        init.append(np.asarray(d["actions"], np.float32))
        raw = rng.normal(size=(30, 32)).astype(np.float32)
        raws.append(raw)
        o = {"state": ns[i].copy(), "actions": raw.copy()}
        o = absolute(unnorm(o))
        outs.append(np.asarray(o["actions"])[:, :23].astype(np.float64))
    W.t("kept", kept.astype(np.float64))
    W.t("init_actions", np.stack(init))
    W.t("raw", np.stack(raws))
    W.t("out_actions", np.stack(outs))
    # cubic compression 26 -> 20 (eval_b1k_wrapper._interpolate_actions)
    acts = rng.normal(size=(6, 26, 23))
    comp = []
    for a in acts:
        oi, ti = np.linspace(0, 25, 26), np.linspace(0, 25, 20)
        c = np.zeros((20, 23))
        for dim in range(23):
            c[:, dim] = interp1d(oi, a[:, dim], kind="cubic")(ti)
        comp.append(c)
    W.t("cubic_in", acts)
    W.t("cubic_out", np.stack(comp))
    # correction rules and gripper variation: states with grippers around the thresholds
    cr_task, cr_stage, cr_state, cr_act, cr_out, cr_ostage, gv = [], [], [], [], [], [], []
    for k in range(400):
        task = int(rng.integers(0, 50))
        stage = int(rng.integers(0, STAGES_2025[task]))
        if k < 40:
            task, stage = 0, int(rng.integers(0, 5))
        s = rng.normal(size=23).astype(np.float32) * 0.5
        s[14] = rng.choice([-1.0, -0.99, -0.5, 0.95, 1.0]).astype(np.float32)
        s[22] = rng.choice([-1.0, -0.985, 0.0, 0.91, 1.0]).astype(np.float32)
        a = rng.normal(size=(30, 23)) * 0.2
        a[:, 14] += rng.choice([0.0, 0.5]) * np.linspace(0, 1, 30)
        o, ostage = apply_correction_rules(task, stage, s, a.copy())
        cr_task.append(task); cr_stage.append(stage); cr_state.append(s); cr_act.append(a)
        cr_out.append(np.asarray(o, np.float64)); cr_ostage.append(ostage)
        gv.append(int(check_gripper_variation(np.asarray(o), 26)[0]))
    W.t("cr_task", np.asarray(cr_task, np.int32)); W.t("cr_stage", np.asarray(cr_stage, np.int32))
    W.t("cr_state", np.stack(cr_state)); W.t("cr_act", np.stack(cr_act))
    W.t("cr_out", np.stack(cr_out)); W.t("cr_ostage", np.asarray(cr_ostage, np.int32))
    W.t("gv", np.asarray(gv, np.int32))
    # stage voting sequences (eval_b1k_wrapper.update_current_stage, history 3, promote 2)
    seqs, res = [], []
    for k in range(200):
        task = int(rng.integers(0, 50))
        n = STAGES_2025[task]
        preds = rng.integers(0, 15, size=40)
        stage, hist, trace = 0, deque(maxlen=3), []
        for p in preds:
            p = min(int(p) if rng.random() < 0.3 else min(stage + int(rng.integers(-1, 3)), 14), 14)
            p = max(p, 0)
            max_stage = n - 1
            pp = min(p, max_stage)
            hist.append(pp)
            if len(hist) == 3:
                nxt = stage + 1
                if nxt <= max_stage:
                    vn = sum(1 for q in hist if q == nxt)
                    vs = sum(1 for q in hist if q == nxt + 1)
                    vb = sum(1 for q in hist if q == stage - 1)
                    if vn >= 2:
                        stage = nxt; hist.clear()
                    elif vs == 3:
                        stage = nxt; hist.clear()
                    elif vb == 3 and stage > 0:
                        stage -= 1; hist.clear()
            seqs.append((task, p))
            trace.append(stage)
        res.append(trace)
    W.t("vote_in", np.asarray(seqs, np.int32).reshape(200, 40, 2))
    W.t("vote_out", np.asarray(res, np.int32))
    W.write(pathlib.Path(args.out), manifest_copy=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
