"""지시 형식 오프라인 실험 — π0.5 추론 접착부 (GPU). 실험 로직·지표는 Rust(src/agent/probe), 여기선 정책 호출만.

표본(<samples>.npz: tools/probe_prep.py)과 prompt 목록(<prompts>.jsonl: `probe prompts`)을 받아 줄마다 policy.infer 를 한 번 부르고
행동 묶음을 <out>.f32 (little-endian float32, [줄 수, H, 23]) 로, 정답 묶음을 <out>_gt.f32 로, 시간·GPU 메모리를 <out>_meta.json 으로 쓴다.
정책이 들어 있는 저장소의 .venv 파이썬으로 실행한다:
  comet : ~/openpi-comet/.venv/bin/python  (입력 키 observation/egocentric_camera·wrist_image_left·wrist_image_right)
  openpi: ~/openpi/.venv/bin/python        (공식 베이스라인, 입력 키 observation/image_0·1·2 = 머리·왼손목·오른손목)
"""

import argparse
import dataclasses
import gc
import json
import time

import numpy as np

KEYS = {
    "comet": ("observation/egocentric_camera", "observation/wrist_image_left", "observation/wrist_image_right"),
    "openpi": ("observation/image_0", "observation/image_1", "observation/image_2"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy_config", required=True)
    ap.add_argument("--policy_dir", required=True)
    ap.add_argument("--input_style", choices=list(KEYS), default="comet")
    ap.add_argument("--repo_id", default=None, help="openpi: 정규화 통계 폴더 이름 (예 turning_on_radio)")
    ap.add_argument("--robot", default=None, help="openpi: 로봇 설정 이름 (예 b1k/R1Pro)")
    args = ap.parse_args()

    import jax

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    # 한 번에 메모리로 (NpzFile 은 data[key] 할 때마다 zip 에서 배열 전체를 다시 읽는다)
    with np.load(args.samples + ".npz") as z:
        data = {k: z[k] for k in z.files}
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    cfg = _config.get_config(args.policy_config)
    if args.repo_id or args.robot:  # 공식 serve_b1k.py 와 같은 방식
        repl = {}
        if args.repo_id:
            repl["repo_id"] = args.repo_id
        if args.robot:
            repl["robot_config_name"] = args.robot
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **repl))
    t = time.monotonic()
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)
    load_s = time.monotonic() - t
    gc.collect()
    kh, kl, kr = KEYS[args.input_style]
    horizon = data["gt"].shape[1]
    preds = np.zeros((len(rows), horizon, 23), np.float32)
    times = []
    for r in rows:
        i = r["sample"]
        batch = {
            kh: data["head"][i],
            kl: data["left"][i],
            kr: data["right"][i],
            "observation/state": data["state"][i],
            "prompt": r["text"],
        }
        t = time.monotonic()
        a = np.asarray(policy.infer(batch)["actions"], np.float32)
        times.append(time.monotonic() - t)
        n = min(horizon, a.shape[0])
        preds[r["row"], :n] = a[:n, :23]
        if r["row"] % 100 == 0:
            print(f"[probe] {r['row']}/{len(rows)} {times[-1] * 1000:.0f} ms", flush=True)
    preds.tofile(args.out + ".f32")
    data["gt"].astype(np.float32).tofile(args.out + "_gt.f32")
    s = jax.local_devices()[0].memory_stats() or {}
    meta = {
        "policy_config": args.policy_config,
        "policy_dir": args.policy_dir,
        "input_style": args.input_style,
        "model_horizon": int(a.shape[0]),
        "rows": len(rows),
        "samples": int(data["gt"].shape[0]),
        "horizon": horizon,
        "load_s": round(load_s, 1),
        "infer_first_s": round(times[0], 2),
        "infer_ms_median": round(1000 * float(np.median(times[1:])), 1) if len(times) > 1 else None,
        "infer_ms_p90": round(1000 * float(np.percentile(times[1:], 90)), 1) if len(times) > 1 else None,
        "jax_bytes_in_use_gib": round(s.get("bytes_in_use", 0) / 2**30, 2),
        "jax_peak_bytes_gib": round(s.get("peak_bytes_in_use", 0) / 2**30, 2),
    }
    json.dump(meta, open(args.out + "_meta.json", "w"), indent=1)
    print("[probe]", json.dumps(meta))


if __name__ == "__main__":
    main()
