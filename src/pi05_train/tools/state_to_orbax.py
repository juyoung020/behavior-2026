"""Offline: native trainer state -> openpi params checkpoint (orbax), so openpi / the export tool can load it.

Takes the base checkpoint's params (everything the trainer did not train), replaces the trained tensors by the ones in
the state file (`p.<path>`, or `ema.<path>` with --ema — openpi saves the EMA params when ema_decay is set,
training/checkpoints.py:150-152) and adds the LoRA factors (mode lora). Result: <out>/params, loadable with
openpi's `_model.restore_params` (a LoRA-mode checkpoint needs the matching LoRA model config).

    JAX_PLATFORMS=cpu wsl_py.sh state_to_orbax.py --state state_step1000.pi05d --base ~/checkpoints/X --out ~/checkpoints/Y [--ema]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--base", required=True, help="checkpoint dir with params/ (the training's starting point)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", action="store_true", help="write the EMA parameters (what openpi saves) instead of raw")
    args = ap.parse_args()
    import orbax.checkpoint as ocp
    from flax import traverse_util
    from make_state import restore_f32
    from train_ref import read_dump

    base = restore_f32(pathlib.Path(args.base).expanduser() / "params")
    st = read_dump(args.state)
    pre = "ema." if args.ema else "p."
    n_rep = n_new = 0
    for k, v in st.items():
        if not k.startswith(pre):
            continue
        name = k[len(pre):]
        if name in base:
            assert base[name].shape == v.shape, (name, base[name].shape, v.shape)
            n_rep += 1
        else:
            n_new += 1  # LoRA factors
        base[name] = np.asarray(v, np.float32)
    tree = traverse_util.unflatten_dict({tuple(k.split("/")): v for k, v in base.items()})
    out = pathlib.Path(args.out).expanduser() / "params"
    with ocp.PyTreeCheckpointer() as ck:
        ck.save(out, {"params": tree})
    print(f"wrote {out}: {n_rep} trained tensors replaced, {n_new} added ({'EMA' if args.ema else 'raw'} params)")


if __name__ == "__main__":
    main()
