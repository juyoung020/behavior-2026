"""시뮬레이터 없이 접착부 → simlink(WSL) 연결을 잰다: 평가기와 같은 키·크기의 관측(RGB-D 래퍼)을 만들어 IntegPolicy 로 보낸다.

    conda activate behavior
    python C:\\behavior-2026\\src\\integ\\glue\\link_bench.py --link 127.0.0.1:7801 --n 600 [--sim-ms 60] [--sync]

- 관측: robot::proprio(61), robot::cam_rel_poses(21), 카메라 3대 RGBA u8 + depth_linear f32 (머리 720², 손목 480²)
- 움직임: 앞 150 스텝 전진(0.4 m/s) → 멈춤(이동 멈춤 경계) → 300 스텝에서 왼 그리퍼 닫기(그리퍼 경계)
- inner 정책은 0 행동(π0.5 없음). --sim-ms 로 시뮬레이터 스텝 시간을 흉내 낸다(보내기 스레드가 그 사이에 보냄).
출력: 스텝별 CSV(접착부 비용: drain_us = 전 ACK 읽기, send_us = 요약·영상 복사·경계면 결정 대기) + 요약.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simlink_policy import IntegPolicy, SimLinkClient  # noqa: E402

K_HEAD = [306.0, 306.0, 360.0, 360.0]
K_WRIST = [388.6639, 388.6639, 240.0, 240.0]


class ZeroPolicy:
    prompt = "turn on the radio"

    def act(self, obs):
        return np.zeros(23, np.float32)

    def reset(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="127.0.0.1:7801")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--sim-ms", type=float, default=60.0)
    ap.add_argument("--task", default="turning_on_radio")
    ap.add_argument("--sync", action="store_true", help="보내기 스레드 없이(비교용)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out or time.strftime("C:/behavior-2026/outputs/integ/link_bench_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    hello = {"task": a.task, "task_id": 0, "num_envs": 1, "hz": 30.0, "max_steps": a.n,
             "cams": [{"name": "head", "w": 720, "h": 720, "k": K_HEAD},
                      {"name": "left_wrist", "w": 480, "h": 480, "k": K_WRIST},
                      {"name": "right_wrist", "w": 480, "h": 480, "k": K_WRIST}],
             "crp_order": ["left_wrist", "right_wrist", "head"], "stage_count": 5, "prompt": ZeroPolicy.prompt,
             "client": "link_bench.py"}
    t0 = time.perf_counter()
    link = SimLinkClient(a.link, hello, async_send=not a.sync)
    print(f"HELLO_ACK {link.hello_ack} ({(time.perf_counter() - t0) * 1e3:.1f} ms)", flush=True)
    keys = [f"robot::robot:{c}:Camera:0::rgb" for c in ("zed_link", "left_realsense_link", "right_realsense_link")]
    pol = IntegPolicy(ZeroPolicy(), link, log_path=str(out / "glue_steps.csv"))
    pol.reset()
    rng = np.random.default_rng(0)
    head = rng.integers(0, 255, (720, 720, 4), dtype=np.uint8)
    wr = rng.integers(0, 255, (480, 480, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:720, 0:720]
    head_d = (1.0 + 0.002 * xx + 0.001 * yy).astype(np.float32)
    wr_d = np.full((480, 480), 0.4, np.float32)
    crp = np.array([0.1, 0.2, 1.0, 0, 0, 0, 1, 0.1, -0.2, 1.0, 0, 0, 0, 1, 0.05, 0.0, 1.6, 0.5, -0.5, 0.5, -0.5], np.float32)
    t_start = time.perf_counter()
    for s in range(a.n):
        p = np.zeros(61, np.float32)
        p[0] = 0.4 if s < 150 else 0.0
        p[2] = 0.3 if 60 <= s < 90 else 0.0
        g = 0.045 if s < 300 else 0.01
        p[24] = p[25] = g
        p[49] = p[50] = 0.045
        obs = {"robot::proprio": p, "robot::cam_rel_poses": crp, "task_id": np.array([0]),
               keys[0]: head, keys[1]: wr, keys[2]: wr,
               keys[0].replace("::rgb", "::depth_linear"): head_d,
               keys[1].replace("::rgb", "::depth_linear"): wr_d,
               keys[2].replace("::rgb", "::depth_linear"): wr_d}
        pol.act(obs)
        if a.sim_ms > 0:
            time.sleep(a.sim_ms / 1e3)
    wall = time.perf_counter() - t_start
    pol.close()
    import csv
    rows = list(csv.DictReader(open(out / "glue_steps.csv", encoding="utf-8")))
    decs = [json.loads(l) for l in open(out / "decisions.jsonl", encoding="utf-8")]

    def stat(v):
        v = np.sort(np.asarray(v, float))
        return f"n {len(v)} p50 {np.percentile(v, 50):.0f} p90 {np.percentile(v, 90):.0f} p99 {np.percentile(v, 99):.0f} max {v.max():.0f}" if len(v) else "n 0"

    hold = [r for r in rows if r["hold"] == "1"]
    fr = [r for r in rows if r["hold"] == "0" and int(r["frames"]) > 0]
    no = [r for r in rows if r["hold"] == "0" and int(r["frames"]) == 0]
    summ = {
        "steps": len(rows), "wall_s": round(wall, 2), "sim_ms": a.sim_ms, "async": not a.sync,
        "holds": len(hold), "frame_steps": len(fr), "plain_steps": len(no),
        "mb_sent": round(sum(int(r["bytes"]) for r in rows) / 1e6, 1),
        "decisions": [(d["step"], d.get("kind"), d.get("trigger"), d.get("text"), d.get("stage"), d.get("decide_ms")) for d in decs if d.get("kind") != "stage"],
        "stage_updates": [(d["step"], d.get("stage")) for d in decs if d.get("kind") == "stage"],
    }
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print("접착부 비용 [µs] (drain = 전 ACK 읽기, send = 요약·영상 복사·큐, 경계면 결정 대기 포함)")
    print("  평소 스텝  drain", stat([float(r["drain_us"]) for r in no]), "| send", stat([float(r["send_us"]) for r in no]))
    print("  영상 스텝  drain", stat([float(r["drain_us"]) for r in fr]), "| send", stat([float(r["send_us"]) for r in fr]))
    print("  경계 스텝  send+결정", stat([float(r["send_us"]) for r in hold]))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("기록:", out)


if __name__ == "__main__":
    main()
