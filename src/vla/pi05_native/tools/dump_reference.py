"""Offline: JAX reference dumps for the layer-by-layer check (JSBSim style, like fdm_verify).

Runs the official openpi pi05 policy (the same objects `scripts/b1k/serve_b1k.py` builds) on a fixed
input set and writes, per sample, one container file (same format as .pi05w, see export_weights.py):

  in.*     raw inputs: 3 x 224x224 uint8 images, 61-dim proprio, prompt bytes, noise
  host.*   what the input transforms hand to the model: token ids, mask, normalized state
  img.*    SigLIP: stem, +posemb, every block output (+ inner points for a few blocks), final tokens
  pre.*    Gemma prefix: every layer output and its K/V cache (+ inner points for a few layers)
  suf.*    action expert: every denoising step x every layer output, v_t, x_t
  out.*    official fused `sample_actions` result (normalized units) and the full `policy.infer` result

Per-layer values come from calling the official flax modules (siglip.Encoder1DBlock, gemma.Block,
gemma.RMSNorm, Pi0.embed_suffix, Pi0.action_out_proj) one layer at a time with the checkpoint params.

--plant DIR: teacher forcing. Every layer takes its input from DIR's dump (the previous layer's output
there) instead of its own chain, so the result isolates one layer's arithmetic. Used to measure the
per-layer noise floor (JAX-CPU layer fed with JAX-GPU inputs vs JAX-GPU).

    JAX_PLATFORMS=cuda wsl_py.sh dump_reference.py --tag gpu --out DIR ...
    JAX_PLATFORMS=cpu  wsl_py.sh dump_reference.py --tag cpu --out DIR ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import ml_dtypes

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_weights import Writer  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
import flax.nnx as nnx  # noqa: E402

from openpi.models import gemma as G  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0 as P  # noqa: E402
from openpi.models import siglip as S  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

BF16 = ml_dtypes.bfloat16
IMG_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
INNER_IMG = (0, 13, 26)
INNER_LLM = (0, 9, 17)
INNER_STEPS = (0, 9)


def read_container(path: pathlib.Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    assert raw[:8] == b"PI05W\0\0\1"
    mlen = int(np.frombuffer(raw[8:16], np.uint64)[0])
    doff = int(np.frombuffer(raw[16:24], np.uint64)[0])
    dt = {"bf16": BF16, "f32": np.float32, "f64": np.float64, "i32": np.int32, "u8": np.uint8}
    out = {}
    for line in raw[24:24 + mlen].decode().splitlines():
        f = line.split()
        if f and f[0] == "t":
            name, d, off, nb, nd = f[1], f[2], int(f[3]), int(f[4]), int(f[5])
            shape = tuple(int(x) for x in f[6:6 + nd])
            out[name] = np.frombuffer(raw, dt[d], count=nb // np.dtype(dt[d]).itemsize, offset=doff + off).reshape(shape)
    return out


def npy(x):
    x = np.asarray(jax.device_get(x))
    return x


def first(inter, *path):
    d = inter
    for p in path:
        d = d[p]
    v = d["__call__"][0]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path.home() / "checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio"))
    ap.add_argument("--asset", default="turning_on_radio")
    ap.add_argument("--config", default="pi05_b1k")
    ap.add_argument("--inputs", default="/mnt/c/behavior-2026/data/pi05_native/inputs_radio.npz")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--tag", required=True, help="file prefix, e.g. gpu / cpu / cpu_planted")
    ap.add_argument("--samples", default="0-31")
    ap.add_argument("--layer-samples", default="0-3", help="samples that get every layer dumped")
    ap.add_argument("--plant", default=None, help="dir with dumps to take every layer input from")
    ap.add_argument("--plant-tag", default="gpu")
    ap.add_argument("--bench", type=int, default=0, help="time N extra policy.infer calls on sample 0")
    args = ap.parse_args()

    def rng_list(s):
        a, b = (s.split("-") + [s])[:2]
        return list(range(int(a), int(b) + 1))

    samples = rng_list(args.samples)
    layer_samples = set(rng_list(args.layer_samples)) if args.layer_samples else set()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.inputs)
    prompt = str(data["prompt"])

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.asset, robot_config_name="b1k/R1Pro"))
    t0 = time.time()
    policy = policy_config.create_trained_policy(cfg, args.ckpt, default_prompt=prompt)
    print(f"[{args.tag}] policy loaded in {time.time() - t0:.1f}s on {jax.devices()}", flush=True)
    model = policy._model
    pure = nnx.state(model).to_pure_dict()
    img_p = pure["PaliGemma"]["img"]
    llm_p = pure["PaliGemma"]["llm"]
    ah = model.action_horizon

    # ---- official modules, called one layer at a time --------------------------------------------
    img_mod = S.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="bfloat16")
    siglip_full = jax.jit(lambda p, im: img_mod.apply({"params": p}, im, train=False))
    enc_block = S.Encoder1DBlock(dtype_mm="bfloat16", mlp_dim=4304, num_heads=16)
    enc_block_j = jax.jit(lambda p, l, x: enc_block.apply({"params": jax.tree.map(lambda a: a[l], p)}, x))
    enc_norm_j = jax.jit(lambda p, x: nn.LayerNorm(dtype="bfloat16").apply({"params": p}, x))
    head_j = jax.jit(lambda p, x: nn.Dense(2048, dtype="bfloat16").apply({"params": p}, x))
    gem = G.Module(configs=[G.get_config("gemma_2b"), G.get_config("gemma_300m")], embed_dtype="bfloat16", adarms=True)
    embed_j = jax.jit(lambda p, t: gem.apply({"params": p}, t, method="embed"))
    block = G.Block(configs=(G.get_config("gemma_2b"), G.get_config("gemma_300m")))
    block_j = jax.jit(lambda p, l, xs, kv, pos, mask, cond: block.apply(
        {"params": jax.tree.map(lambda a: a[l], p)}, xs, kv, pos, mask, cond, capture_intermediates=True, mutable=["intermediates"]))
    rms = G.RMSNorm()
    final_norm_j = jax.jit(lambda p, x, c: rms.apply({"params": p}, x, c))
    # pi0.py:159-186 (pi05 branch) on the three small nnx.Linear modules only: nnx.jit of the whole model would
    # copy all 6.7 GB of weights in and out of the jitted call.
    def _embed_suffix(ain, tin, tout, x_t, t):
        action_tokens = ain(x_t)
        time_emb = P.posemb_sincos(t, ain.out_features, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(tout(nnx.swish(tin(time_emb))))
        return action_tokens, time_emb

    embed_suffix_j = nnx.jit(_embed_suffix)
    out_proj_j = nnx.jit(lambda lin, x: lin(x))
    update_j = jax.jit(lambda x, v, t, dt: (x + dt * v, t + dt), static_argnums=3)

    # layer params are sliced inside jit (no second copy of the weights on the device)
    blk_img = img_p["Transformer"]["encoderblock"]
    blk_llm = llm_p["layers"]

    bench = {}
    for i in samples:
        full = i in layer_samples
        W = Writer()
        planted = None
        if args.plant:
            planted = read_container(pathlib.Path(args.plant) / f"{args.plant_tag}_{i:03d}.pi05d")

        def P_(name, own):
            """teacher forcing: take `name` from the planted dump when present."""
            if planted is not None and name in planted:
                return jnp.asarray(planted[name])
            return own

        def rec(name, x, always=False):
            if full or always:
                W.t(name, np.ascontiguousarray(npy(x)))

        imgs = data["images"][i]
        obs = {"observation/image_0": imgs[0], "observation/image_1": imgs[1], "observation/image_2": imgs[2],
               "observation/state": data["proprio"][i], "prompt": prompt}
        noise = data["noise"][i]
        W.t("in.img", imgs)
        W.t("in.proprio", data["proprio"][i].astype(np.float32))
        W.t("in.prompt", np.frombuffer(prompt.encode(), np.uint8))
        W.t("in.noise", noise)

        # official end-to-end (input transforms -> fused sample_actions -> output transforms)
        res = policy.infer(obs, noise=noise)
        W.t("out.actions", np.asarray(res["actions"], np.float64))
        inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
        W.t("host.tokens", np.asarray(inputs["tokenized_prompt"], np.int32))
        W.t("host.mask", np.asarray(inputs["tokenized_prompt_mask"], np.uint8))
        W.t("host.state", np.asarray(inputs["state"], np.float32))
        W.t("host.state_f64", np.asarray(inputs["state"], np.float64))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        raw = policy._sample_actions(jax.random.key(0), observation, noise=jnp.asarray(noise)[None])
        W.t("out.actions_raw", npy(raw)[0].astype(np.float32))
        observation = _model.preprocess_observation(None, observation, train=False)

        # ---- SigLIP ----------------------------------------------------------------------------
        img_tokens = []
        for c, key in enumerate(IMG_KEYS):
            x_off, out = siglip_full(img_p, observation.images[key])
            rec(f"img.c{c}.stem", out["stem"][0].reshape(256, 1152), always=True)
            rec(f"img.c{c}.posemb", out["with_posemb"][0], always=True)
            rec(f"img.c{c}.tokens_official", x_off[0], always=True)
            x = P_(f"img.c{c}.posemb", out["with_posemb"][0])[None].astype(jnp.bfloat16)
            for l in range(27):
                y, o = enc_block_j(blk_img, l, x)
                if l in INNER_IMG:
                    rec(f"img.c{c}.l{l}.sa", o["sa"][0]); rec(f"img.c{c}.l{l}.res1", o["+sa"][0])
                    rec(f"img.c{c}.l{l}.mlp", o["mlp"][0])
                rec(f"img.c{c}.l{l}.out", y[0])
                x = P_(f"img.c{c}.l{l}.out", y[0])[None]
            enc = enc_norm_j(img_p["Transformer"]["encoder_norm"], x)
            rec(f"img.c{c}.encoded", enc[0])
            enc = P_(f"img.c{c}.encoded", enc[0])[None]
            tok = head_j(img_p["head"], enc)
            rec(f"img.c{c}.tokens", tok[0], always=True)
            img_tokens.append(P_(f"img.c{c}.tokens", tok[0])[None])

        # ---- prefix -------------------------------------------------------------------------------
        txt = embed_j(llm_p, observation.tokenized_prompt)
        rec("txt.emb", txt[0], always=True)
        txt = P_("txt.emb", txt[0])[None]
        prefix = jnp.concatenate(img_tokens + [txt], axis=1)
        prefix_mask = jnp.concatenate([jnp.ones((1, 768), bool), observation.tokenized_prompt_mask], axis=1)
        attn_mask = P.make_attn_mask(prefix_mask, jnp.zeros(prefix.shape[1], bool))
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        x = prefix.astype(jnp.bfloat16)
        kv_cache = []
        for l in range(18):
            (xs, (k, v)), inter = block_j(blk_llm, l, [x, None], None, positions, attn_mask[:, None], [None, None])
            y = xs[0]
            it = inter["intermediates"]
            rec(f"pre.l{l}.norm1", first(it, "pre_attention_norm")[0][0])
            rec(f"pre.l{l}.norm2", first(it, "pre_ffw_norm")[0][0])
            rec(f"pre.l{l}.q", first(it, "attn", "q_einsum")[0])
            rec(f"pre.l{l}.kv", first(it, "attn", "kv_einsum")[:, 0])
            if l in INNER_LLM:
                rec(f"pre.l{l}.attn", first(it, "attn", "attn_vec_einsum")[0])
                rec(f"pre.l{l}.mlp", first(it, "mlp")[0])
            rec(f"pre.l{l}.out", y[0])
            rec(f"pre.l{l}.k", k[0, :, 0], always=(l in (0, 17)))
            rec(f"pre.l{l}.v", v[0, :, 0], always=(l in (0, 17)))
            kv_cache.append((P_(f"pre.l{l}.k", k[0, :, 0])[None, :, None], P_(f"pre.l{l}.v", v[0, :, 0])[None, :, None]))
            x = P_(f"pre.l{l}.out", y[0])[None]

        # ---- denoising loop -----------------------------------------------------------------------
        dt = -1.0 / 10
        x_t = jnp.asarray(noise)[None]
        t = jnp.asarray(1.0, jnp.float32)
        step = 0
        while float(t) >= -dt / 2:
            s = f"suf.s{step}"
            W.t(f"{s}.time", np.asarray([float(t)], np.float32))
            x_t = P_(f"suf.s{step - 1}.x", x_t[0])[None] if step > 0 else x_t
            tokens, cond = embed_suffix_j(model.action_in_proj, model.time_mlp_in, model.time_mlp_out, x_t,
                                          jnp.broadcast_to(t, (1,)))
            smask = jnp.ones((1, ah), bool)
            sar = jnp.array([True] + [False] * (ah - 1))
            rec(f"{s}.cond", cond[0], always=(step == 0))
            rec(f"{s}.in", tokens[0], always=(step == 0))
            sattn = P.make_attn_mask(smask, sar)
            pattn = jnp.broadcast_to(prefix_mask[:, None, :], (1, ah, prefix_mask.shape[1]))
            full_mask = jnp.concatenate([pattn, sattn], axis=-1)
            spos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
            h = tokens.astype(jnp.bfloat16)
            for l in range(18):
                (xs, _), inter = block_j(blk_llm, l, [None, h], kv_cache[l], spos, full_mask[:, None], [None, cond])
                y = xs[1]
                it = inter["intermediates"]
                rec(f"{s}.l{l}.norm1", first(it, "pre_attention_norm_1")[0][0])
                rec(f"{s}.l{l}.norm2", first(it, "pre_ffw_norm_1")[0][0])
                if l in INNER_LLM and step in INNER_STEPS:
                    rec(f"{s}.l{l}.gate1", first(it, "pre_attention_norm_1")[1][0, 0])
                    rec(f"{s}.l{l}.attn", first(it, "attn", "attn_vec_einsum_1")[0])
                    rec(f"{s}.l{l}.mlp", first(it, "mlp_1")[0])
                rec(f"{s}.l{l}.out", y[0])
                h = P_(f"{s}.l{l}.out", y[0])[None]
            fin, _ = final_norm_j(llm_p["final_norm_1"], h, cond)
            rec(f"{s}.final", fin[0])
            fin = P_(f"{s}.final", fin[0])[None]
            v_t = out_proj_j(model.action_out_proj, fin[:, -ah:])
            rec(f"{s}.v", v_t[0], always=True)
            v_t = P_(f"{s}.v", v_t[0])[None]
            x_t, t = update_j(x_t, v_t, t, dt)
            rec(f"{s}.x", x_t[0], always=True)
            step += 1
        W.t("out.actions_chain", npy(x_t)[0].astype(np.float32))
        d = np.abs(npy(x_t)[0] - npy(raw)[0]).max()
        W.write(out_dir / f"{args.tag}_{i:03d}.pi05d", manifest_copy=False)
        print(f"[{args.tag}] sample {i}: steps={step} |chain - official| max {d:.3g}", flush=True)

    if args.bench:
        obs0 = {"observation/image_0": data["images"][0][0], "observation/image_1": data["images"][0][1],
                "observation/image_2": data["images"][0][2], "observation/state": data["proprio"][0], "prompt": prompt}
        for _ in range(3):
            policy.infer(obs0, noise=data["noise"][0])
        ts, ms = [], []
        for _ in range(args.bench):
            a = time.perf_counter()
            r = policy.infer(obs0, noise=data["noise"][0])
            ts.append((time.perf_counter() - a) * 1e3)
            ms.append(r["policy_timing"]["infer_ms"])
        stats = {}
        try:
            stats = jax.devices()[0].memory_stats() or {}
        except Exception:  # noqa: BLE001
            pass
        bench = {"infer_total_ms_median": float(np.median(ts)), "model_ms_median": float(np.median(ms)),
                 "infer_total_ms_min": float(np.min(ts)), "n": args.bench,
                 "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
                 "bytes_in_use": int(stats.get("bytes_in_use", 0)), "device": str(jax.devices()[0])}
        (out_dir / f"{args.tag}_bench.json").write_text(json.dumps(bench, indent=1))
        print(f"[{args.tag}] bench {bench}", flush=True)


if __name__ == "__main__":
    main()
