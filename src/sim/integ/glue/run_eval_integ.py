"""통합 한 판 실행기: 공식 평가기(omnigibson.eval.eval, 무수정) --policy local 의 LocalPolicy 에
네이티브 π0.5(Pi05NativePolicy, 평가기 프로세스 안) + 계획기·scenemap 연결(IntegPolicy → simlink, WSL)을 넣는다.

    python run_eval_integ.py --weights W.pi05w --link 127.0.0.1:7801 --out <폴더> [--wrapper rgbd|default]
        [--robot-config C:/behavior-2026/src/sim/configs/r1pro_openpi.yaml] [--prompt …] [--replan 16] [--seed 0]
        [--stage-count N] [--no-link] -- <평가기 인자, 예: --task-name turning_on_radio --max-steps 600 …>

src\\pi05_native\\glue\\run_eval_native.py 와 같은 방식(런타임에 LocalPolicy.__init__ 만 바꿈, BEHAVIOR-1K 무수정)이고,
정책을 IntegPolicy 로 한 겹 감싼 것만 다르다. π0.5 쪽 파일(pi05_policy.py)은 고치지 않는다.
"""
import argparse
import json
import os
import pathlib
import runpy
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
NATIVE = HERE.parents[2] / "vla" / "pi05_native"
sys.path.insert(0, str(NATIVE / ("build_win" if os.name == "nt" else "build")))
sys.path.insert(0, str(NATIVE / "glue"))
sys.path.insert(0, str(HERE))


def crp_order(robot_config: str):
    """cam_rel_poses 순서 = 로봇 설정 eval.camera_sensor_names 순서(평가기 robot_camera_names.values())."""
    import yaml

    cfg = yaml.safe_load(open(robot_config, encoding="utf-8"))
    names = (cfg.get("eval") or {}).get("camera_sensor_names") or {}
    return list(names.keys()) or ["left_wrist", "right_wrist", "head"]


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pi05w 경로, 또는 pb2025(2025 1위 모델, 과제별 체크포인트)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--replan", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", default="127.0.0.1:7801")
    ap.add_argument("--no-link", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrapper", default="rgbd", choices=["rgbd", "default"])
    ap.add_argument("--robot-config", default="C:/behavior-2026/src/sim/configs/r1pro_openpi.yaml")
    ap.add_argument("--stage-count", type=int, default=-1, help="-1 = 엔진이 알려 준 값, 없으면 2025 표")
    ap.add_argument("--no-apply-stage", action="store_true")
    ap.add_argument("--no-apply-prompt", action="store_true")
    args = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--policy" in eval_args:
        i = eval_args.index("--policy")
        del eval_args[i:i + 2]
    task = eval_args[eval_args.index("--task-name") + 1]
    max_steps = int(eval_args[eval_args.index("--max-steps") + 1]) if "--max-steps" in eval_args else None
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from run_eval_native import task_prompt
    prompt = args.prompt or task_prompt(task)

    import _pi05native as N
    from pi05_policy import Pi05NativePolicy, pb2025_weights_for_task
    from simlink_policy import IntegPolicy, SimLinkClient, camera_specs, stage_count_for
    import omnigibson.eval.policies as P
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES

    task_id = TASK_NAMES_TO_INDICES.get(task)
    orig_init = P.LocalPolicy.__init__
    holder = {}

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        weights = pb2025_weights_for_task if args.weights == "pb2025" else args.weights  # 1위 모델: 과제별 체크포인트
        inner = Pi05NativePolicy(weights, prompt, replan_every=args.replan, log_path=str(out / "native_steps.csv"),
                                 seed=args.seed)
        link = None
        if not args.no_link:
            sc = args.stage_count if args.stage_count >= 0 else stage_count_for(task_id, getattr(inner, "info", None))
            hello = {"task": task, "task_id": task_id, "num_envs": 1, "hz": 30.0, "max_steps": max_steps,
                     "cams": camera_specs(args.wrapper), "crp_order": crp_order(args.robot_config), "stage_count": sc,
                     "prompt": prompt, "client": f"run_eval_integ.py pid {os.getpid()}"}
            t0 = time.perf_counter()
            link = SimLinkClient(args.link, hello)
            print(f"[integ] simlink 연결 {(time.perf_counter() - t0) * 1e3:.0f} ms: {link.hello_ack} stage_count={sc}", flush=True)
            (out / "hello.json").write_text(json.dumps({"hello": hello, "ack": link.hello_ack}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        self.policy = IntegPolicy(inner, link, native=N, log_path=str(out / "glue_steps.csv"),
                                  apply_prompt=not args.no_apply_prompt, apply_stage=not args.no_apply_stage)
        holder["p"] = self.policy

    P.LocalPolicy.__init__ = init
    wrapper = {"rgbd": "omnigibson.eval.wrappers.RGBDFullResWrapper", "default": "omnigibson.eval.wrappers.DefaultWrapper"}
    if "--env-wrapper" in eval_args:
        i = eval_args.index("--env-wrapper")
        del eval_args[i:i + 2]
    sys.argv = ["omnigibson.eval.eval", *eval_args, "--env-wrapper", wrapper[args.wrapper], "--policy", "local"]
    print(f"[integ] 평가기 인자: {sys.argv[1:]}", flush=True)
    try:
        runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)
    finally:
        if "p" in holder:
            holder["p"].close()


if __name__ == "__main__":
    main()
