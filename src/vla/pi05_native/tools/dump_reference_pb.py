"""Offline: JAX reference dumps for the PiBehavior (2025 BEHAVIOR 1st place) model, same container format as
dump_reference.py. Uses the alstar8 fork (refs/behavior-1k-solution_alstar8, commit 5d57037) on top of the WSL openpi
venv, config `pi_behavior_2025_submission`, and the official `create_trained_policy` / `PiBehaviorPolicy`.

Per sample (radio demo frames, task 0, stage = sample % 5):
  host.*    state tokens / normalized state
  pre.x0    prefix embeddings [805, 2048] (images + 5 task tokens + 32 state tokens)
  pre.l*.out  prefix layer outputs (official gemma.Block, one layer at a time, prefix attention mask)
  pre.stage_logits, pre.kv2.l{0,17}.{k,v}  stage head and layer-mixed KV cache
  suf.s*.v / suf.s*.x   every denoising step (20), per-layer suffix calls with the mixed cache
  out.actions_raw  fused official sample_actions with explicit noise (no inpainting)
  out.actions      policy.infer with the same noise (robot units)
  inp.*     a second call with initial_actions (rolling inpainting) through policy.infer and the policy RNG:
            the engine reproduces the JAX key stream, so this checks correlated noise + inpainting end to end.

    OPENPI_BEHAVIOR_DATA_ROOT=/mnt/c/behavior-2026/data JAX_PLATFORMS=cuda wsl_py.sh dump_reference_pb.py --tag gpu
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys
import time

sys.path.insert(0, "/mnt/c/behavior-2026/refs/behavior-1k-solution_alstar8/src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("OPENPI_BEHAVIOR_DATA_ROOT", "/mnt/c/behavior-2026/data")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.nnx as nnx  # noqa: E402

from export_weights import Writer  # noqa: E402
from openpi.models import gemma as G  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from b1k.models.observation import Observation, preprocess_observation  # noqa: E402
from b1k.policies import policy_config as _pc  # noqa: E402
from b1k.training import config as _config  # noqa: E402


def npy(x):
    return np.asarray(jax.device_get(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path.home() / "checkpoints/behavior_submission/checkpoint_2"))
    ap.add_argument("--inputs", default="/mnt/c/behavior-2026/data/pi05_native/inputs_radio.npz")
    ap.add_argument("--out", default="/mnt/c/behavior-2026/data/pi05_native/ref_pb")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", default="0-3")
    ap.add_argument("--layers", default="0-1", help="samples with every layer dumped")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    rng_list = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[-1]) + 1))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.inputs)

    cfg = _config.get_config("pi_behavior_2025_submission")
    t0 = time.time()
    policy = _pc.create_trained_policy(cfg, args.ckpt, sample_kwargs={"num_steps": 20})
    print(f"[{args.tag}] policy loaded in {time.time() - t0:.1f}s on {jax.devices()}", flush=True)
    model = policy._model
    pure = nnx.state(model).to_pure_dict()
    llm_p = pure["PaliGemma"]["llm"]
    blk = llm_p["layers"]
    ah = model.action_horizon
    block = G.Block(configs=(G.get_config("gemma_2b"), G.get_config("gemma_300m")))
    block_j = jax.jit(lambda p, l, xs, kv, pos, mask, cond: block.apply(
        {"params": jax.tree.map(lambda a: a[l], p)}, xs, kv, pos, mask, cond))
    rms = G.RMSNorm()
    final_norm_j = jax.jit(lambda p, x, c: rms.apply({"params": p}, x, c))

    def _embed_suffix(ain, tin, tout, x_t, t):
        from openpi.models.pi0 import posemb_sincos
        a = ain(x_t)
        te = posemb_sincos(t, ain.out_features, min_period=4e-3, max_period=4.0)
        te = nnx.swish(tout(nnx.swish(tin(te))))
        return a, te

    embed_suffix_j = nnx.jit(_embed_suffix)
    out_proj_j = nnx.jit(lambda lin, x: lin(x))
    update_j = jax.jit(lambda x, v, t, dt: (x + dt * v, t + dt), static_argnums=3)

    for i in rng_list(args.samples):
        full = i in set(rng_list(args.layers))
        W = Writer()
        imgs = data["images"][i]
        proprio = data["proprio"][i].astype(np.float32)
        task, stage = 0, i % 5
        noise = data["noise"][i][:ah].astype(np.float32)  # [30, 32] standard normal, used as given
        obs = {"observation/egocentric_camera": imgs[0], "observation/wrist_image_left": imgs[1],
               "observation/wrist_image_right": imgs[2], "observation/state": proprio,
               "tokenized_prompt": np.array([task, stage], np.int32),
               "tokenized_prompt_mask": np.array([True, True]), "subtask_state": np.array(stage, np.int32)}
        W.t("in.img", imgs)
        W.t("in.proprio", proprio)
        W.t("in.noise", noise)
        W.t("in.task_stage", np.array([task, stage], np.int32))

        res = policy.infer(dict(obs), noise=noise)
        W.t("out.actions", np.asarray(res["actions"], np.float64))
        W.t("out.stage_logits_official", np.asarray(res["subtask_logits"], np.float32))
        inputs = policy._input_transform(jax.tree.map(lambda x: x, dict(obs)))
        W.t("host.state", np.asarray(inputs["state"], np.float32))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = Observation.from_dict(batched)
        raw, logits = policy._sample_actions(jax.random.key(0), observation, noise=jnp.asarray(noise)[None])
        W.t("out.actions_raw", npy(raw)[0].astype(np.float32))
        W.t("out.stage_logits", npy(logits)[0].astype(np.float32))

        # ---- per-layer prefix / suffix -----------------------------------------------------------------
        ob = preprocess_observation(None, observation, train=False)
        prefix, pmask, par = model.embed_prefix(ob)
        W.t("pre.x0", npy(prefix[0]).astype(ml_bf16()))
        amask = make_attn_mask(pmask, par)
        pos = jnp.cumsum(pmask, axis=1) - 1
        x = prefix.astype(jnp.bfloat16)
        ks, vs = [], []
        for l in range(18):
            (xs, (k, v)) = block_j(blk, l, [x, None], None, pos, amask[:, None], [None, None])
            x = xs[0]
            if full:
                W.t(f"pre.l{l}.out", npy(x[0]))
            ks.append(k)
            vs.append(v)
        fin, _ = final_norm_j(llm_p["final_norm"], x, None)
        base_idx = int(np.argmax(np.asarray(par))) - 1
        W.t("pre.base_final", npy(fin[0, base_idx]))
        kv = (jnp.stack(ks), jnp.stack(vs))
        kv2 = model.kv_transform(kv)
        for l in (0, 17):
            W.t(f"pre.kv2.l{l}.k", npy(kv2[0][l, 0, :, 0]))
            W.t(f"pre.kv2.l{l}.v", npy(kv2[1][l, 0, :, 0]))
        dt = -1.0 / 20
        x_t = jnp.asarray(noise)[None]
        t = jnp.asarray(1.0, jnp.float32)
        step = 0
        while float(t) >= -dt / 2:
            tokens, cond = embed_suffix_j(model.action_in_proj, model.time_mlp_in, model.time_mlp_out, x_t,
                                          jnp.broadcast_to(t, (1,)))
            smask = jnp.ones((1, ah), bool)
            sattn = make_attn_mask(smask, jnp.array([True] + [False] * (ah - 1)))
            full_mask = jnp.concatenate([jnp.broadcast_to(pmask[:, None, :], (1, ah, pmask.shape[1])), sattn], axis=-1)
            spos = jnp.sum(pmask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
            h = tokens.astype(jnp.bfloat16)
            for l in range(18):
                (xs, _) = block_j(blk, l, [None, h], (kv2[0][l], kv2[1][l]), spos, full_mask[:, None], [None, cond])
                h = xs[1]
            fin, _ = final_norm_j(llm_p["final_norm_1"], h, cond)
            v_t = out_proj_j(model.action_out_proj, fin[:, -ah:])
            x_t, t = update_j(x_t, v_t, t, dt)
            W.t(f"suf.s{step}.v", npy(v_t[0]))
            W.t(f"suf.s{step}.x", npy(x_t[0]).astype(np.float32))
            step += 1
        W.t("out.actions_chain", npy(x_t)[0].astype(np.float32))

        # ---- inpainting + correlated noise through the policy RNG ------------------------------------------
        policy._rng = jax.random.key(1234 + i)
        # openpi DeltaActions subtracts the state from "actions" in place, so every consumer gets its own copy
        kept = np.asarray(res["actions"], np.float64)[26:30].copy()
        r2 = policy.infer(dict(obs), initial_actions=kept.copy())
        W.t("inp.kept", kept)  # robot units, before the delta transform
        W.t("inp.seed", np.array([1234 + i], np.int32))
        W.t("inp.actions", np.asarray(r2["actions"], np.float64))
        W.t("inp.stage_logits", np.asarray(r2["subtask_logits"], np.float32))
        # the pieces of that call, recomputed with the same keys (pi_behavior_policy.py:34-95, pi_behavior.py:957-966)
        _, sample_rng = jax.random.split(jax.random.key(1234 + i))
        _, noise_rng = jax.random.split(sample_rng)
        W.t("inp.noise", npy(model.generate_correlated_noise(noise_rng, 1))[0].astype(np.float32))
        tb = policy._input_transform({**dict(obs), "actions": kept.copy()})
        x0 = np.asarray(tb["actions"], np.float32)
        W.t("inp.x0", x0)
        raw_i, _ = policy._sample_actions(sample_rng, observation, initial_actions=jnp.asarray(x0)[None])
        W.t("inp.raw", npy(raw_i)[0].astype(np.float32))
        raw_n, _ = policy._sample_actions(sample_rng, observation)  # same key, no inpainting
        W.t("inp.raw_noinp", npy(raw_n)[0].astype(np.float32))
        W.write(out_dir / f"{args.tag}_{i:03d}.pi05d", manifest_copy=False)
        print(f"[{args.tag}] sample {i}: stage {stage} steps {step} |chain-official| "
              f"{np.abs(npy(x_t)[0] - npy(raw)[0]).max():.3g}", flush=True)

    if args.bench:
        d0 = np.load(args.inputs)
        obs0 = {"observation/egocentric_camera": d0["images"][0][0], "observation/wrist_image_left": d0["images"][0][1],
                "observation/wrist_image_right": d0["images"][0][2],
                "observation/state": d0["proprio"][0].astype(np.float32),
                "tokenized_prompt": np.array([0, 0], np.int32), "tokenized_prompt_mask": np.array([True, True]),
                "subtask_state": np.array(0, np.int32)}
        for _ in range(3):
            policy.infer(dict(obs0))
        ts = []
        for _ in range(args.bench):
            a = time.perf_counter()
            policy.infer(dict(obs0))
            ts.append((time.perf_counter() - a) * 1e3)
        print(f"[{args.tag}] bench infer median {np.median(ts):.1f} ms min {np.min(ts):.1f} ms", flush=True)


def ml_bf16():
    import ml_dtypes

    return ml_dtypes.bfloat16


if __name__ == "__main__":
    main()
