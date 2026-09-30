"""Offline: JAX reference for the training-step randomness and augmentation (native trainer check).

For openpi's key chain (scripts/train.py: rng = key(seed); train_rng, _ = split(rng); per step fold_in(train_rng, step);
models/pi0.py:192-197 split(., 3) -> preprocess / noise / time) this dumps, for a few steps:
  the keys, noise = normal(noise_rng, [B, 32, 32]), time = beta(time_rng, 1.5, 1, [B]) * 0.999 + 0.001,
  and preprocess_observation(preprocess_rng, obs, train=True) images (augmax crop/resize/rotate + color jitter).

    JAX_PLATFORMS=cuda wsl_py.sh aug_ref.py --tag gpu
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, "/mnt/c/behavior-2026/src/pi05_native/tools")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/mnt/c/behavior-2026/data/pi05_native/ref")
    ap.add_argument("--out", default="/mnt/c/behavior-2026/data/pi05_train/ref")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", default="0,1,7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from export_weights import Writer
    from train_ref import read_dump

    B = args.batch
    ds = [read_dump(pathlib.Path(args.ref) / f"gpu_{i:03d}.pi05d") for i in range(B)]
    img_u8 = np.stack([d["in.img"] for d in ds])
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = _model.Observation(
        images={k: jnp.asarray(img_u8[:, c].astype(np.float32) / 255.0 * 2.0 - 1.0) for c, k in enumerate(keys)},
        image_masks={k: jnp.ones((B,), bool) for k in keys},
        state=jnp.zeros((B, 32), jnp.float32))
    rng = jax.random.key(args.seed)
    train_rng, _ = jax.random.split(rng)
    W = Writer()
    W.c("seed", args.seed); W.c("batch", B); W.c("steps", args.steps)
    W.t("in.img", img_u8)
    for s in [int(x) for x in args.steps.split(",")]:
        r = jax.random.fold_in(train_rng, s)
        pre, nk, tk = jax.random.split(r, 3)
        noise = jax.random.normal(nk, (B, 32, 32))
        time = jax.random.beta(tk, 1.5, 1, (B,)) * 0.999 + 0.001
        o = _model.preprocess_observation(pre, obs, train=True)
        W.t(f"s{s}.key", np.asarray(jax.random.key_data(r), np.uint32).view(np.int32))
        W.t(f"s{s}.noise", np.asarray(noise, np.float32))
        W.t(f"s{s}.time", np.asarray(time, np.float32))
        W.t(f"s{s}.img", np.stack([np.asarray(o.images[k], np.float32) for k in keys], axis=1))  # [B][3][224][224][3]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W.write(out / f"aug_{args.tag}.pi05d", manifest_copy=False)
    print(f"[{args.tag}] wrote aug reference for steps {args.steps}")


if __name__ == "__main__":
    main()
