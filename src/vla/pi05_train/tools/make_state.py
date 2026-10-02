"""Offline: initial training state for the native trainer from an openpi orbax checkpoint (full depth).

Writes the trainable parameters as f32 in openpi naming/shape (`p.<path>`), plus the optimizer / schedule / EMA
settings of the training config (default `pi05_b1k`, scripts/train.py + training/optimizer.py). The frozen weights are
read by the trainer from the inference-format model file (`pi05_native/tools/export_weights.py`), so they are not
repeated here. LoRA factors (mode lora) are initialised like openpi's LoRAConfig.init_fn (normal, stddev 0.01) from a
numpy generator (same distribution, not the same numbers as a JAX init).

    JAX_PLATFORMS=cpu wsl_py.sh make_state.py --ckpt ~/checkpoints/pi05_base/params-parent --mode expert --out X.pi05d
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, "/mnt/c/behavior-2026/src/vla/pi05_native/tools")
import numpy as np  # noqa: E402

EXPERT = re.compile(r".*llm.*_1.*|action_in_proj/.*|action_out_proj/.*|time_mlp_in/.*|time_mlp_out/.*")


def restore_f32(params_dir):
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import traverse_util
    import jax

    with ocp.PyTreeCheckpointer() as ckptr:
        meta = ckptr.metadata(params_dir)
        item = {"params": meta["params"]}
        p = ckptr.restore(params_dir, ocp.args.PyTreeRestore(
            item=item, restore_args=jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=jnp.float32), item)))["params"]
    flat = traverse_util.flatten_dict(p)
    if all(k[-1] == "value" for k in flat):
        flat = {k[:-1]: v for k, v in flat.items()}
    return {"/".join(k): np.asarray(v, np.float32) for k, v in flat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing params/")
    ap.add_argument("--mode", choices=["expert", "lora"], default="expert")
    ap.add_argument("--config", default="pi05_b1k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from export_weights import Writer
    from openpi.training import config as _config

    tcfg = _config.get_config(args.config)
    p = restore_f32(pathlib.Path(args.ckpt).expanduser() / "params")
    W = Writer()
    W.c("mode", args.mode); W.c("img.depth", 27); W.c("llm.depth", 18); W.c("config", args.config)
    lr, o = tcfg.lr_schedule, tcfg.optimizer
    W.c("lr.warmup", lr.warmup_steps); W.c("lr.peak", repr(lr.peak_lr)); W.c("lr.decay_steps", lr.decay_steps)
    W.c("lr.end", repr(lr.decay_lr))
    W.c("adam.b1", o.b1); W.c("adam.b2", o.b2); W.c("adam.eps", repr(o.eps)); W.c("adam.wd", repr(o.weight_decay))
    W.c("clip", o.clip_gradient_norm); W.c("ema", tcfg.ema_decay); W.c("batch", tcfg.batch_size)
    W.c("num_train_steps", tcfg.num_train_steps)
    import ml_dtypes

    n = 0
    for k in sorted(list(p.keys())):
        v = p.pop(k)  # drop the f32 copy as we go (peak memory)
        train = EXPERT.fullmatch(k) is not None if args.mode == "expert" else (
            k.startswith("PaliGemma/img/") or not k.startswith("PaliGemma/llm/"))
        if train:
            W.t("p." + k, v)
            n += v.size
        elif args.mode == "lora" and not k.endswith("input_embedding"):
            W.t("f." + k, v.astype(ml_dtypes.bfloat16))  # frozen params in bf16 (scripts/train.py:103-104)
    if args.mode == "lora":
        # LoRA factors (gemma.py:88-107 ranks 16 / 32; lora.py LoRAConfig.init_fn = normal(0.01) for a and b)
        rng = np.random.default_rng(args.seed)
        L = 18
        for sfx, D, Fd, r in (("", 2048, 16384, 16), ("_1", 1024, 4096, 32)):
            shapes = {
                f"attn/q_einsum{sfx}/lora_a": (L, 8, D, r), f"attn/q_einsum{sfx}/lora_b": (L, 8, r, 256),
                f"attn/kv_einsum{sfx}/lora_a": (L, 2, 1, D, r), f"attn/kv_einsum{sfx}/lora_b": (L, 2, 1, r, 256),
                f"attn/attn_vec_einsum{sfx}/lora_a": (L, 8, 256, r), f"attn/attn_vec_einsum{sfx}/lora_b": (L, 8, r, D),
                f"mlp{sfx}/gating_einsum_lora_a": (L, 2, D, r), f"mlp{sfx}/gating_einsum_lora_b": (L, 2, r, Fd),
                f"mlp{sfx}/linear_lora_a": (L, Fd, r), f"mlp{sfx}/linear_lora_b": (L, r, D),
            }
            for name, shp in shapes.items():
                v = (rng.standard_normal(shp) * 0.01).astype(np.float32)
                W.t("p.PaliGemma/llm/layers/" + name, v)
                n += v.size
    W.write(pathlib.Path(args.out), manifest_copy=False)
    print(f"wrote {args.out}: {n / 1e6:.1f} M trainable params ({args.mode})")


if __name__ == "__main__":
    main()
