"""Offline: JAX reference for the native pi0.5 trainer (openpi `pi05_b1k` training step), same container format as
the inference dumps (`pi05_native/tools/export_weights.py` Writer).

What is reproduced (openpi `behavior` branch in WSL ~/openpi):
  loss      models/pi0.py:189-214 compute_loss, with the noise / flow time passed in instead of drawn
            (x_t = t*noise + (1-t)*a, u = noise - a, loss = mean over (batch, horizon) of mean_d (v - u)^2),
            preprocess_observation(train=False) (no augmentation; augmentation is checked separately)
  params    scripts/train.py:85-117 init_train_state: frozen params cast to bf16, trainable stay f32
  step      scripts/train.py:137-180 train_step: value_and_grad over the trainable filter, optax chain
            (clip_by_global_norm 1.0 -> adamw b1 .9 b2 .95 eps 1e-8 wd 1e-10, warmup-cosine lr), EMA 0.99
  modes     expert: action expert (llm *_1) + action_in/out_proj + time_mlp trainable, the rest frozen
            lora:   Pi0Config(paligemma_variant=gemma_2b_lora, action_expert_variant=gemma_300m_lora) and its
                    get_freeze_filter (SigLIP fully trainable, Gemma/expert only through LoRA)
The model can be cut to fewer layers (--img-depth / --llm-depth; the first layers of the real checkpoint) so the
reference fits in memory; the native trainer reads the same file and runs the same cut model.

Outputs (DIR/<tag>_*.pi05d):
  <tag>_state.pi05d   params (openpi names, '/'-joined; f32 trainable, bf16 frozen), batch, noise, time, config
  <tag>_grad.pi05d    loss, per-(sample, step) loss, grads of every trainable param (f32), global grad norm
  <tag>_opt.pi05d     params + EMA after optimizer steps 1 and N (--steps), loss per step

    JAX_PLATFORMS=cuda wsl_py.sh train_ref.py --mode expert --img-depth 2 --llm-depth 2 --batch 4 --tag gpu
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, "/mnt/c/behavior-2026/src/vla/pi05_native/tools")

import numpy as np  # noqa: E402

TRAINABLE_EXPERT = r".*llm.*_1.*|action_in_proj/.*|action_out_proj/.*|time_mlp_in/.*|time_mlp_out/.*"


def read_dump(path):
    import ml_dtypes

    b = pathlib.Path(path).read_bytes()
    ml = int(np.frombuffer(b[8:16], np.uint64)[0])
    do = int(np.frombuffer(b[16:24], np.uint64)[0])
    dt = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "f64": np.float64, "i32": np.int32, "u8": np.uint8}
    out = {}
    for ln in b[24:24 + ml].decode().splitlines():
        f = ln.split()
        if not f or f[0] != "t":
            continue
        off, nb, nd = int(f[3]), int(f[4]), int(f[5])
        out[f[1]] = np.frombuffer(b[do + off:do + off + nb], dt[f[2]]).reshape([int(x) for x in f[6:6 + nd]])
    return out


def model_name(a):
    o = f"_o{a.img_offset}-{a.llm_offset}" if (a.img_offset or a.llm_offset) else ""
    return f"model_i{a.img_depth}_l{a.llm_depth}{o}.pi05w"


def patch_depths(img_depth, llm_depth):
    from openpi.models import gemma as G
    from openpi.models import siglip as S

    og, ov = G.get_config, S.decode_variant
    G.get_config = lambda v: dataclasses.replace(og(v), depth=llm_depth)
    S.decode_variant = lambda v: {**ov(v), "depth": img_depth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path.home() / "checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio"))
    ap.add_argument("--ref", default="/mnt/c/behavior-2026/data/pi05_native/ref", help="inference dumps (batch source)")
    ap.add_argument("--out", default="/mnt/c/behavior-2026/data/pi05_train/ref")
    ap.add_argument("--mode", choices=["expert", "lora"], default="expert")
    ap.add_argument("--img-depth", type=int, default=2)
    ap.add_argument("--llm-depth", type=int, default=2)
    ap.add_argument("--img-offset", type=int, default=0, help="first SigLIP layer of the cut (real checkpoint layers)")
    ap.add_argument("--llm-offset", type=int, default=0, help="first Gemma layer of the cut")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry", action="store_true", help="print the trainable parameter list (shapes only) and exit")
    args = ap.parse_args()

    patch_depths(args.img_depth, args.llm_depth)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    from flax import traverse_util
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared import nnx_utils
    from openpi.training import optimizer as _opt
    from openpi.training import config as _config
    from export_weights import Writer, restore_bf16

    if args.mode == "lora":
        mcfg = pi0_config.Pi0Config(pi05=True, action_horizon=32, paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora")
        freeze = mcfg.get_freeze_filter()
    else:
        mcfg = pi0_config.Pi0Config(pi05=True, action_horizon=32)
        freeze = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(TRAINABLE_EXPERT)))
    trainable = nnx.All(nnx.Param, nnx.Not(freeze))
    tcfg = _config.get_config("pi05_b1k")  # optimizer / lr schedule / ema exactly as the training config

    if args.dry:
        model = nnx.eval_shape(lambda: mcfg.create(jax.random.key(0)))
        from flax import traverse_util
        n = 0
        for k, v in traverse_util.flatten_dict(nnx.state(model, trainable).to_pure_dict()).items():
            print("/".join(map(str, k)), tuple(v.shape))
            n += int(np.prod(v.shape))
        print(f"trainable params: {n / 1e6:.2f} M; optimizer {tcfg.optimizer}; lr {tcfg.lr_schedule}; ema {tcfg.ema_decay}")
        return

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # ---- params: the real checkpoint (bf16 as stored for serving -> f32), first layers only --------------------------
    flat = restore_bf16(pathlib.Path(args.ckpt).expanduser() / "params")
    part, cut = {}, {}
    for k, v in flat.items():
        if k.startswith("PaliGemma/img/Transformer/encoderblock/"):
            v = v[args.img_offset:args.img_offset + args.img_depth]
        elif k.startswith("PaliGemma/llm/layers/"):
            v = v[args.llm_offset:args.llm_offset + args.llm_depth]
        cut[k] = v
        part[tuple(k.split("/"))] = np.asarray(v, np.float32)
    del flat
    # the same cut model in the inference engine format (the native trainer runs the frozen prefix with it)
    from export_weights import export_backbone
    M = Writer()
    M.c("model", "pi05"); M.c("action_dim", 32); M.c("action_horizon", 32); M.c("max_token_len", 200)
    M.c("num_steps", 10)
    M.c("img.width", 1152); M.c("img.depth", args.img_depth); M.c("img.mlp", 4304); M.c("img.mlp_pad", 4352)
    M.c("img.heads", 16); M.c("img.head_dim", 72); M.c("img.patch", 14); M.c("img.res", 224)
    M.c("llm.width", 2048); M.c("llm.depth", args.llm_depth); M.c("llm.mlp", 16384); M.c("llm.heads", 8)
    M.c("llm.kv_heads", 1); M.c("llm.head_dim", 256); M.c("llm.vocab", 257152)
    M.c("ae.width", 1024); M.c("ae.mlp", 4096)
    export_backbone(cut, M, args.img_depth, args.llm_depth)
    M.write(out_dir / model_name(args), manifest_copy=False)
    del cut, M
    model = mcfg.create(jax.random.key(args.seed))  # LoRA factors keep this random init (normal 0.01)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(part))
    model = nnx.merge(graphdef, state)
    params = nnx.state(model)
    params = nnx_utils.state_map(params, freeze, lambda p: p.replace(p.value.astype(jnp.bfloat16)))  # train.py:103-104
    nnx.update(model, params)
    print(f"[{args.tag}] model ready in {time.time() - t0:.1f}s on {jax.devices()}", flush=True)

    # ---- batch: the radio demo frames of the inference reference (same images, tokens, state) -----------------------
    B = args.batch
    ds = [read_dump(pathlib.Path(args.ref) / f"gpu_{i:03d}.pi05d") for i in range(B)]
    img_u8 = np.stack([d["in.img"] for d in ds])  # [B, 3, 224, 224, 3]
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = _model.Observation(
        images={k: jnp.asarray(img_u8[:, c].astype(np.float32) / 255.0 * 2.0 - 1.0) for c, k in enumerate(keys)},
        image_masks={k: jnp.ones((B,), bool) for k in keys},
        state=jnp.asarray(np.stack([d["host.state"] for d in ds]).astype(np.float32)),
        tokenized_prompt=jnp.asarray(np.stack([d["host.tokens"] for d in ds]).astype(np.int32)),
        tokenized_prompt_mask=jnp.asarray(np.stack([d["host.mask"] for d in ds]).astype(bool)),
    )
    actions = np.stack([d["out.actions_raw"] for d in ds]).astype(np.float32)  # normalized model-space chunks
    rs = np.random.default_rng(args.seed)
    noise = rs.standard_normal((args.steps, B, 32, 32)).astype(np.float32)
    ftime = (rs.beta(1.5, 1.0, (args.steps, B)) * 0.999 + 0.001).astype(np.float32)

    def per_sample_loss(m, noise, t):  # pi0.py:189-214 with noise/time given
        o = _model.preprocess_observation(None, obs, train=False)
        te = t[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        pt, pm, par = m.embed_prefix(o)
        st_, sm, sar, cond = m.embed_suffix(o, x_t, t)
        im = jnp.concatenate([pm, sm], axis=1)
        am = make_attn_mask(im, jnp.concatenate([par, sar], axis=0))
        pos = jnp.cumsum(im, axis=1) - 1
        (_, so), _ = m.PaliGemma.llm([pt, st_], mask=am, positions=pos, adarms_cond=[None, cond])
        v_t = m.action_out_proj(so[:, -m.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B, ah]

    diff = nnx.DiffState(0, trainable)

    @nnx.jit
    def grad_fn(m, noise, t):
        def lf(m):
            ps = per_sample_loss(m, noise, t)
            return jnp.mean(ps), ps
        (loss, ps), g = nnx.value_and_grad(lf, argnums=diff, has_aux=True)(m)
        return loss, ps, g

    def flat_state(s):
        return {"/".join(map(str, k)): np.asarray(jax.device_get(v))
                for k, v in traverse_util.flatten_dict(s.to_pure_dict()).items()}

    # state file
    W = Writer()
    W.c("mode", args.mode); W.c("img.depth", args.img_depth); W.c("llm.depth", args.llm_depth)
    W.c("img.offset", args.img_offset); W.c("llm.offset", args.llm_offset); W.c("model_file", model_name(args))
    W.c("batch", B); W.c("steps", args.steps)
    lr = tcfg.lr_schedule
    W.c("lr.warmup", lr.warmup_steps); W.c("lr.peak", repr(lr.peak_lr)); W.c("lr.decay_steps", lr.decay_steps)
    W.c("lr.end", repr(lr.decay_lr))
    o = tcfg.optimizer
    W.c("adam.b1", o.b1); W.c("adam.b2", o.b2); W.c("adam.eps", repr(o.eps)); W.c("adam.wd", repr(o.weight_decay))
    W.c("clip", o.clip_gradient_norm); W.c("ema", tcfg.ema_decay)
    tr_names = set(flat_state(nnx.state(model, trainable)))
    for k, v in flat_state(nnx.state(model, nnx.Param)).items():
        if k not in tr_names and k.endswith("input_embedding"):
            continue  # frozen token table (1 GB) lives in the model file
        W.t(("p." if k in tr_names else "f.") + k, v)
    W.t("in.img", img_u8)
    W.t("in.tokens", np.stack([d["host.tokens"] for d in ds]).astype(np.int32))
    W.t("in.mask", np.stack([d["host.mask"] for d in ds]).astype(np.uint8))
    W.t("in.state", np.stack([d["host.state"] for d in ds]).astype(np.float32))
    W.t("in.actions", actions)
    W.t("in.noise", noise)
    W.t("in.time", ftime)
    W.write(out_dir / f"{args.tag}_state.pi05d", manifest_copy=False)

    # gradient at the initial params (step 0 noise/time)
    loss, ps, g = grad_fn(model, jnp.asarray(noise[0]), jnp.asarray(ftime[0]))
    G = Writer()
    G.t("loss", np.array([float(loss)], np.float32))
    G.t("loss_ps", np.asarray(ps, np.float32))
    G.t("grad_norm", np.array([float(optax.global_norm(g))], np.float32))
    for k, v in flat_state(g).items():
        G.t("g." + k, v.astype(np.float32))
    G.write(out_dir / f"{args.tag}_grad.pi05d", manifest_copy=False)
    print(f"[{args.tag}] loss {float(loss):.6f} grad_norm {float(optax.global_norm(g)):.6f}", flush=True)

    # optimizer steps (train.py:137-180), step s uses noise[s], time[s]
    tx = _opt.create_optimizer(tcfg.optimizer, tcfg.lr_schedule, weight_decay_mask=None)
    p0 = nnx.state(model, trainable)
    opt_state = tx.init(p0)
    ema = nnx.state(model, nnx.Param) if tcfg.ema_decay is not None else None

    @nnx.jit
    def step_fn(m, opt_state, noise, t):
        loss, _, g = grad_fn(m, noise, t)
        p = nnx.state(m, trainable)
        upd, opt_state = tx.update(g, opt_state, p)
        nnx.update(m, optax.apply_updates(p, upd))
        return loss, opt_state

    O = Writer()
    losses = []
    for s in range(args.steps):
        loss, opt_state = step_fn(model, opt_state, jnp.asarray(noise[s]), jnp.asarray(ftime[s]))
        losses.append(float(loss))
        new = nnx.state(model, nnx.Param)
        if ema is not None:
            ema = jax.tree.map(lambda a, b: tcfg.ema_decay * a + (1 - tcfg.ema_decay) * b, ema, new)
        if s + 1 in (1, args.steps):
            for k, v in flat_state(nnx.state(model, trainable)).items():
                O.t(f"s{s + 1}.p.{k}", v)
            if ema is not None:
                ema_flat = flat_state(ema)
                for k in tr_names:
                    O.t(f"s{s + 1}.ema.{k}", ema_flat[k])
    O.t("loss_steps", np.array(losses, np.float32))
    O.write(out_dir / f"{args.tag}_opt.pi05d", manifest_copy=False)
    print(f"[{args.tag}] step losses {' '.join(f'{x:.5f}' for x in losses)}  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
