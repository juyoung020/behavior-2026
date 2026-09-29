"""One-off offline tool: openpi pi05 orbax checkpoint -> our raw weight file (.pi05w).

Runs in the WSL openpi venv (reads orbax/JAX). The engine itself never imports Python.

    cd ~/openpi && JAX_PLATFORMS=cpu .venv/bin/python \
        /mnt/c/behavior-2026/src/pi05_native/tools/export_weights.py \
        --ckpt ~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio \
        --asset turning_on_radio --out /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w

File layout (little endian):
    bytes 0..7    magic  b"PI05W\\0\\0\\1"
    bytes 8..15   u64    manifest byte length (text, UTF-8)
    bytes 16..23  u64    data offset (multiple of 4096)
    manifest      text lines:
                    "cfg <key> <value...>"                 model/config scalars and small vectors
                    "t <name> <dtype> <offset> <nbytes> <ndim> <d0> <d1> ..."   tensors
                  offsets are relative to the data offset, each tensor starts on a 256-byte boundary.
    data          raw tensors

Weights are cast to bfloat16 exactly the way the openpi server loads them
(`_model.restore_params(..., dtype=jnp.bfloat16)`, openpi src/openpi/policies/policy_config.py:57),
i.e. orbax does the f32->bf16 cast (round to nearest even). Linear weights are stored transposed as
[out, in] (in contiguous) so both GEMM operands are K-major; the MLP gate/up matrices are row-interleaved
(row 2j = gate j, row 2j+1 = up j) so one GEMM epilogue can form gelu(gate)*up. SigLIP MLP hidden size
4304 is zero-padded to 4352 (a multiple of 64): zero weight rows/bias give gelu(0)=0 and zero K columns
add exact zeros, so results are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16


def restore_bf16(params_dir: pathlib.Path) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import traverse_util

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_dir)
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            params_dir,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax_tree_map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=jnp.bfloat16), item
                ),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return {"/".join(k): np.asarray(v) for k, v in flat.items()}


def jax_tree_map(f, tree):
    import jax

    return jax.tree.map(f, tree)


class Writer:
    def __init__(self):
        self.tensors: list[tuple[str, np.ndarray]] = []
        self.cfg: list[str] = []

    def t(self, name: str, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        assert arr.dtype in (BF16, np.float32, np.float64, np.int32, np.uint8), (name, arr.dtype)
        self.tensors.append((name, arr))

    def c(self, key: str, *vals):
        self.cfg.append("cfg " + key + " " + " ".join(str(v) for v in vals))

    def write(self, path: pathlib.Path, manifest_copy: bool = True):
        dt_name = {np.dtype(BF16): "bf16", np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
                   np.dtype(np.int32): "i32", np.dtype(np.uint8): "u8"}
        lines = list(self.cfg)
        off = 0
        layout = []
        for name, arr in self.tensors:
            off = (off + 255) // 256 * 256
            layout.append((name, arr, off))
            lines.append(f"t {name} {dt_name[arr.dtype]} {off} {arr.nbytes} {arr.ndim} " + " ".join(map(str, arr.shape)))
            off += arr.nbytes
        manifest = ("\n".join(lines) + "\n").encode()
        data_off = (24 + len(manifest) + 4095) // 4096 * 4096
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        with open(tmp, "wb") as f:
            f.write(b"PI05W\0\0\1")
            f.write(np.uint64(len(manifest)).tobytes())
            f.write(np.uint64(data_off).tobytes())
            f.write(manifest)
            f.write(b"\0" * (data_off - 24 - len(manifest)))
            pos = 0
            for name, arr, o in layout:
                if o > pos:
                    f.write(b"\0" * (o - pos))
                f.write(arr.tobytes())
                pos = o + arr.nbytes
        tmp.replace(path)
        if manifest_copy:
            (path.with_suffix(".manifest.txt")).write_bytes(manifest)
        return off


def T(w):  # [in, out] -> [out, in]
    return np.ascontiguousarray(np.asarray(w).T)


def interleave_rows(gate_t, up_t):  # both [H, K] -> [2H, K]
    h, k = gate_t.shape
    out = np.empty((2 * h, k), gate_t.dtype)
    out[0::2] = gate_t
    out[1::2] = up_t
    return out


def pad_rows(a, n):
    out = np.zeros((n,) + a.shape[1:], a.dtype)
    out[: a.shape[0]] = a
    return out


def pad_cols(a, n):
    out = np.zeros(a.shape[:-1] + (n,), a.dtype)
    out[..., : a.shape[-1]] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing params/ and assets/")
    ap.add_argument("--asset", required=True, help="asset id holding norm_stats.json, e.g. turning_on_radio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="paligemma_tokenizer.model (default: openpi download cache)")
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--robot", default="b1k/R1Pro")
    ap.add_argument("--arch", default="pi05", choices=["pi05", "pi_behavior"],
                    help="pi_behavior = 2025 1st place (IliaLarchenko/behavior-1k-solution) PiBehavior model")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt).expanduser()
    p = restore_bf16(ckpt / "params")
    print(f"restored {len(p)} arrays", file=sys.stderr)

    W = Writer()
    if args.arch == "pi_behavior":
        export_pi_behavior(args, ckpt, p, W)
        export_backbone(p, W)
        total = W.write(pathlib.Path(args.out))
        print(f"wrote {args.out}: {len(W.tensors)} tensors, {total / 2**30:.2f} GiB", file=sys.stderr)
        return
    # ---- config -------------------------------------------------------------------------------
    # Pi0Config(action_horizon=32, pi05=True): openpi src/openpi/training/config.py:758,
    # widths/depths from src/openpi/models/gemma.py:69-87 and siglip.py:311-370 ("So400m/14").
    W.c("model", "pi05")
    W.c("action_dim", 32)
    W.c("action_horizon", args.action_horizon)
    W.c("max_token_len", 200)
    W.c("num_steps", 10)
    W.c("img.width", 1152); W.c("img.depth", 27); W.c("img.mlp", 4304); W.c("img.mlp_pad", 4352)
    W.c("img.heads", 16); W.c("img.head_dim", 72); W.c("img.patch", 14); W.c("img.res", 224)
    W.c("llm.width", 2048); W.c("llm.depth", 18); W.c("llm.mlp", 16384); W.c("llm.heads", 8)
    W.c("llm.kv_heads", 1); W.c("llm.head_dim", 256); W.c("llm.vocab", 257152)
    W.c("ae.width", 1024); W.c("ae.mlp", 4096)
    W.c("source", str(ckpt))

    # ---- robot / normalization (host side) -------------------------------------------------------
    sys.path.insert(0, str(pathlib.Path.home() / "openpi" / "src"))
    from openpi.configs.robots import ROBOT_REGISTRY
    from openpi.shared import normalize as _normalize
    from openpi.training.config import LeRobotB1KDataConfig

    robot = ROBOT_REGISTRY[args.robot]
    W.c("robot.name", robot.name)
    W.c("robot.action_dim", robot.action_dim)
    for i, key in enumerate(sorted(robot.observations.keys())):  # eval_b1k_wrapper.py:78 sorted()
        W.c(f"robot.cam{i}", robot.observations[key].obs_key)
    prop = []
    for pc in robot.proprio:  # b1k_policy.py:22-41
        prop.append(("sum" if pc.is_eef else "idx") + ":" + ",".join(map(str, pc.indices)))
    W.c("robot.proprio", *prop)
    maps = LeRobotB1KDataConfig(robot_config_name=args.robot)._build_delta_mappings(robot)
    for a_idx, s_idx in maps:  # MappedAbsoluteActions, transforms.py:272-290
        W.c("robot.delta", ",".join(map(str, a_idx)) + ":" + ",".join(map(str, s_idx)))
    grip = [i for ac in robot.action if ac.is_eef for i in ac.indices]
    W.c("robot.grippers", *grip)

    ns = _normalize.load(ckpt / "assets" / args.asset)  # z-score (use_quantile_norm=False, config.py:455)
    for key in ("state", "actions"):
        W.t(f"norm.{key}.mean", np.asarray(ns[key].mean, np.float64))
        W.t(f"norm.{key}.std", np.asarray(ns[key].std, np.float64))

    tok_path = args.tokenizer
    if tok_path is None:
        from openpi.shared import download

        tok_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    tok = pathlib.Path(tok_path).read_bytes()
    W.t("tokenizer.model", np.frombuffer(tok, np.uint8))
    W.c("tokenizer.sha256", hashlib.sha256(tok).hexdigest())
    export_backbone(p, W)
    total = W.write(pathlib.Path(args.out))
    print(f"wrote {args.out}: {len(W.tensors)} tensors, {total / 2**30:.2f} GiB", file=sys.stderr)


# 2025 1st place stage table (TASK_NUM_STAGES_2025, alstar8 pi_behavior_config.py:62-68 = original
# IliaLarchenko pi_behavior_config.py:29-35)
TASK_NUM_STAGES_2025 = (
    5, 6, 15, 15, 14, 12, 9, 15, 10, 15,
    7, 13, 10, 15, 15, 15, 15, 11, 13, 12,
    14, 15, 9, 15, 15, 15, 15, 15, 15, 15,
    11, 10, 10, 13, 5, 5, 14, 6, 8, 10,
    5, 15, 8, 15, 12, 11, 9, 14, 15, 15,
)


def export_pi_behavior(args, ckpt, p, W):
    """PiBehavior (behavior-1k-solution/src/b1k/models/pi_behavior.py) extras + host-side tables.
    Config = alstar8 `pi_behavior_2025_submission` (training/config.py:428-472): horizon 30, 50 tasks, 2025 stage
    table, correlated noise beta 0.5, kv transform, per-timestamp action z-score, delta mask (-3,3,-1,7,-1,7,-1)."""
    import json

    W.c("model", "pi_behavior")
    W.c("action_dim", 32)
    W.c("action_horizon", 30)
    W.c("max_token_len", 0)
    W.c("num_steps", 20)  # serve_b1k.py --num-steps 20
    W.c("img.width", 1152); W.c("img.depth", 27); W.c("img.mlp", 4304); W.c("img.mlp_pad", 4352)
    W.c("img.heads", 16); W.c("img.head_dim", 72); W.c("img.patch", 14); W.c("img.res", 224)
    W.c("llm.width", 2048); W.c("llm.depth", 18); W.c("llm.mlp", 16384); W.c("llm.heads", 8)
    W.c("llm.kv_heads", 1); W.c("llm.head_dim", 256); W.c("llm.vocab", 257152)
    W.c("ae.width", 1024); W.c("ae.mlp", 4096)
    W.c("source", str(ckpt))
    W.c("pb.num_tasks", 50)
    W.c("pb.stages", *TASK_NUM_STAGES_2025)
    W.c("pb.inpaint_threshold", 0.3)
    W.c("pb.beta", 0.5)
    # 2026 eval robot (r1pro.yaml name robot_r1) and 61-d compact proprio (alstar8 b1k_proprio.py)
    W.c("robot.name", "robot_r1")
    W.c("robot.action_dim", 23)
    for i, k in enumerate(("zed_link", "left_realsense_link", "right_realsense_link")):
        W.c(f"robot.cam{i}", f"robot_r1::robot_r1:{k}:Camera:0::rgb")
    W.c("robot.kind", "pi_behavior")
    # delta mask make_bool_mask(-3, 3, -1, 7, -1, 7, -1) (training/config.py LeRobotB1KDataConfig)
    mask = [0] * 3 + [1] * 3 + [0] + [1] * 7 + [0] + [1] * 7 + [0]
    W.c("pb.delta_mask", *mask)

    ns = json.loads((ckpt / "assets" / args.asset / "norm_stats.json").read_text())
    ns = ns.get("norm_stats", ns)
    W.t("norm.state.mean", np.asarray(ns["state"]["mean"], np.float64))
    W.t("norm.state.std", np.asarray(ns["state"]["std"], np.float64))
    W.t("norm.actions.mean", np.asarray(ns["actions"]["mean"], np.float64))
    W.t("norm.actions.std", np.asarray(ns["actions"]["std"], np.float64))
    W.t("norm.actions.pt_mean", np.asarray(ns["actions"]["per_timestamp_mean"], np.float64))  # [30, 32]
    W.t("norm.actions.pt_std", np.asarray(ns["actions"]["per_timestamp_std"], np.float64))

    # correlated noise: L_reg = chol(beta * L L^T + (1 - beta) I) (pi_behavior.py:335-355), L from norm stats as f32
    L = np.asarray(ns["actions"]["action_correlation_cholesky"], np.float32).astype(np.float64)
    beta = 0.5
    sig = beta * (L @ L.T) + (1 - beta) * np.eye(L.shape[0])
    Lr = np.linalg.cholesky(sig)
    W.t("pb.corr_L", Lr)  # f64 [960, 960]
    # inpainting correction for the standard wrapper (keep 4 actions, all 32 dims): Sigma_UO Sigma_OO^-1
    # (pi_behavior.py:403-450; Sigma = L_reg L_reg^T)
    S = Lr @ Lr.T
    O = np.arange(4 * 32)
    U = np.arange(4 * 32, 30 * 32)
    soo = S[np.ix_(O, O)]
    eps = 1e-6 * max(np.mean(np.diag(soo)), 1.0)
    C = np.linalg.solve(soo + eps * np.eye(len(O)), S[np.ix_(U, O)].T).T
    W.t("pb.inpaint_C4", C.astype(np.float32))  # [832, 128]

    W.t("pb.task_emb", p["task_embeddings/embedding"])  # [50, 2048]
    W.t("pb.task_stage_emb", p["task_stage_embeddings/embedding"])  # [596, 1024]
    for n in ("gate_sincos", "gate_task_stage", "gate_task", "fusion_layer1", "fusion_layer2", "stage_projection",
              "stage_pred_from_vlm"):
        W.t(f"pb.{n}.w", T(p[f"{n}/kernel"]))  # [out, in]
        W.t(f"pb.{n}.b", p[f"{n}/bias"])
    for n in ("k_coeffs", "v_coeffs", "k_bias", "v_bias"):
        W.t(f"pb.kv.{n}", p[f"kv_transform/{n}"])  # bf16, as restored


def export_backbone(p, W):
    # ---- SigLIP So400m/14 (PaliGemma/img) ---------------------------------------------------------
    g = lambda k: p["PaliGemma/img/" + k]
    emb = g("embedding/kernel")  # [14,14,3,1152] HWIO; patch vector order (kh, kw, c)
    W.t("img.patch_w", T(emb.reshape(14 * 14 * 3, 1152)))  # [1152, 588]
    W.t("img.patch_b", g("embedding/bias"))
    W.t("img.pos", g("pos_embedding")[0])  # [256, 1152]
    E = "Transformer/encoderblock/"
    for l in range(27):
        a = lambda k: g(E + k)[l]
        W.t(f"img.l{l}.ln1_s", a("LayerNorm_0/scale")); W.t(f"img.l{l}.ln1_b", a("LayerNorm_0/bias"))
        qkv = [a(f"MultiHeadDotProductAttention_0/{n}/kernel").reshape(1152, 1152) for n in ("query", "key", "value")]
        W.t(f"img.l{l}.qkv_w", np.concatenate([T(x) for x in qkv], 0))  # [3456, 1152]
        W.t(f"img.l{l}.qkv_b", np.concatenate(
            [a(f"MultiHeadDotProductAttention_0/{n}/bias").reshape(1152) for n in ("query", "key", "value")]))
        W.t(f"img.l{l}.o_w", T(a("MultiHeadDotProductAttention_0/out/kernel").reshape(1152, 1152)))
        W.t(f"img.l{l}.o_b", a("MultiHeadDotProductAttention_0/out/bias"))
        W.t(f"img.l{l}.ln2_s", a("LayerNorm_1/scale")); W.t(f"img.l{l}.ln2_b", a("LayerNorm_1/bias"))
        W.t(f"img.l{l}.fc1_w", pad_rows(T(a("MlpBlock_0/Dense_0/kernel")), 4352))  # [4352, 1152]
        W.t(f"img.l{l}.fc1_b", pad_cols(a("MlpBlock_0/Dense_0/bias"), 4352))
        W.t(f"img.l{l}.fc2_w", pad_cols(T(a("MlpBlock_0/Dense_1/kernel")), 4352))  # [1152, 4352]
        W.t(f"img.l{l}.fc2_b", a("MlpBlock_0/Dense_1/bias"))
    W.t("img.post_s", g("Transformer/encoder_norm/scale")); W.t("img.post_b", g("Transformer/encoder_norm/bias"))
    W.t("img.head_w", T(g("head/kernel"))); W.t("img.head_b", g("head/bias"))  # [2048, 1152]

    # ---- Gemma 2B (expert 0) and Gemma 300M action expert (expert 1) ------------------------------
    L = lambda k: p["PaliGemma/llm/" + k]
    W.t("llm.embed", L("embedder/input_embedding"))  # [257152, 2048], host-side gather
    W.t("llm.final_norm", L("final_norm/scale"))
    W.t("ae.final_mod_w", T(L("final_norm_1/Dense_0/kernel"))); W.t("ae.final_mod_b", L("final_norm_1/Dense_0/bias"))
    for l in range(18):
        a = lambda k: L("layers/" + k)[l]
        for pre, sfx, width, heads in (("llm", "", 2048, 8), ("ae", "_1", 1024, 8)):
            q = a(f"attn/q_einsum{sfx}/w")  # [N, D, H]
            q_t = np.transpose(q, (0, 2, 1)).reshape(heads * 256, width)  # rows (head, h)
            kv = a(f"attn/kv_einsum{sfx}/w")[:, 0]  # [2, D, H]
            kv_t = np.transpose(kv, (0, 2, 1)).reshape(2 * 256, width)
            W.t(f"{pre}.l{l}.qkv_w", np.concatenate([q_t, kv_t], 0))  # [N*H + 2H, D]
            o = a(f"attn/attn_vec_einsum{sfx}/w")  # [N, H, D]
            W.t(f"{pre}.l{l}.o_w", T(o.reshape(heads * 256, width)))  # [D, N*H]
            gu = a(f"mlp{sfx}/gating_einsum")  # [2, D, F]
            W.t(f"{pre}.l{l}.gu_w", interleave_rows(T(gu[0]), T(gu[1])))  # [2F, D]
            W.t(f"{pre}.l{l}.down_w", T(a(f"mlp{sfx}/linear")))  # [D, F]
        W.t(f"llm.l{l}.attn_norm", a("pre_attention_norm/scale"))
        W.t(f"llm.l{l}.ffn_norm", a("pre_ffw_norm/scale"))
        W.t(f"ae.l{l}.attn_mod_w", T(a("pre_attention_norm_1/Dense_0/kernel")))  # [3072, 1024]
        W.t(f"ae.l{l}.attn_mod_b", a("pre_attention_norm_1/Dense_0/bias"))
        W.t(f"ae.l{l}.ffn_mod_w", T(a("pre_ffw_norm_1/Dense_0/kernel")))
        W.t(f"ae.l{l}.ffn_mod_b", a("pre_ffw_norm_1/Dense_0/bias"))

    # ---- pi0.5 heads (nnx.Linear, pi0.py:92-100) --------------------------------------------------
    for n in ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"):
        W.t(f"{n}.w", T(p[f"{n}/kernel"]))  # [out, in]
        W.t(f"{n}.b", p[f"{n}/bias"])


if __name__ == "__main__":
    main()
