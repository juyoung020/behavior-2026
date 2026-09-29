"""포팅 평가기 진입점 — 공식 평가기와 같은 인자·같은 결과 JSON, 시뮬레이터만 이 엔진 (docs/엔진_자체구현.md 10.1·10.2).

    python src/engine/eval/ported_eval.py --backend {engine,dummy} [--scene-root DIR] -- <공식 omnigibson.eval.eval 인자 그대로>
    예) ... ported_eval.py --backend dummy -- --task-name turning_on_radio --robot-config src/configs/r1pro_openpi.yaml \
            --host 127.0.0.1 --port 8011 --instance-indices 0 --num-envs 1 --max-steps 5 --output-dir /tmp/ported

원리
- 공식 omnigibson/eval/eval.py 의 main() 을 그대로 부른다(인자 해석, 시드, 인스턴스 번호, 설정, JSON 폴더, 요약).
  그 안에서 만드는 BatchedEvaluator 만 PortedEvaluator 로 바꿔 끼운다(이 프로세스 안에서 이름만 바꿈, 원본 파일 무수정).
- PortedEvaluator 는 공식 BatchedEvaluator 를 상속하고, 시뮬레이터에 닿는 메서드만 엔진 겉모습(facade.EngineEnv)으로 덮는다.
  판 묶음 루프·정책 웹소켓·지표(AgentMetric/TaskMetric)·결과 JSON·관측 전처리는 공식 코드 그대로다.
- Kit/Isaac Sim 을 띄우지 않는다(og.launch 안 부름). og.sim 자리에는 엔진의 EngineSim 을 둔다.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

logger = logging.getLogger("ported_eval")


def _split_args(argv):
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else ([], argv)
    p = argparse.ArgumentParser(description="포팅 평가기 (엔진 백엔드)")
    p.add_argument("--backend", choices=("engine", "dummy"), default="engine")
    p.add_argument("--scene-root", default=os.path.expanduser("~/engine-data/scenes"),
                   help="원본에서 뽑은 장면 추출물 (과제/인스턴스별, 저장소 밖)")
    p.add_argument("--instrument", action="append", default=[],
                   help="tools/eval_instrumented.py 선택지를 그대로 (예: --instrument=--trace). 원본 평가기와 같은 trace.npz")
    return p.parse_args(ours), rest


def make_ported_evaluator(E, backend: str, scene_root: str):
    import omnigibson as og
    import torch as th
    from hydra.utils import get_class, instantiate
    from gello.utils.og_teleop_utils import load_available_tasks
    from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER
    from omnigibson.eval.utils.score_utils import load_human_stats

    from facade import load_backend

    class PortedEvaluator(E.BatchedEvaluator):
        """공식 BatchedEvaluator 에서 시뮬레이터에 닿는 곳만 엔진으로."""

        def load_env(self, env_wrapper):
            # 공식 load_env(evaluator.py:240) 중 설정 만들기 부분은 같은 함수로, og.Environment 만 엔진으로
            task_name = self.cfg.task.name
            available_tasks = load_available_tasks()
            assert task_name in available_tasks, f"Got invalid task name: {task_name}"
            self.human_stats = load_human_stats(task_name)
            task_cfg = available_tasks[task_name][0]
            robot_cfg = self._build_robot_config(task_name=task_name, task_cfg=task_cfg)  # 공식 (robot_eval_config 도 채움)
            self.robot_name = robot_cfg["name"]
            camera_spec = get_class(env_wrapper["_target_"]).camera_spec()
            if self.cfg.max_steps is None:
                max_steps = int(self.human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)
            else:
                max_steps = int(self.cfg.max_steps)
            env = load_backend(
                backend, task_name=task_name, task_cfg=task_cfg, robot_cfg=robot_cfg,
                robot_eval_cfg=self.robot_eval_config, camera_spec=camera_spec, max_steps=max_steps,
                num_envs=self.num_envs, mode=self.cfg.get("mode", "public_test"),
                partial_scene_load=bool(self.cfg.partial_scene_load), scene_root=scene_root)
            og.sim = env.sim  # TaskMetric.reset 이 og.sim.get_rendering_dt(), __init__ 끝이 og.sim.update_handles()
            og.shutdown = lambda *a, **k: None  # 공식 __exit__ 끝의 og.shutdown(): Kit 을 띄운 적이 없어 할 일 없음
            env._eval_robot_config = self.robot_eval_config
            return instantiate(env_wrapper, env=env)  # 공식 래퍼 그대로 (DefaultWrapper 는 통과만)

        @property
        def should_sync_lights(self) -> bool:
            # 빛 켜기 동기화(LIGHT_EVAL_TASKS, 현재 turning_out_all_lights_before_sleep 하나)는 렌더 전용 — 물리·판정과 무관.
            # 렌더 모듈이 붙으면 엔진이 빛 상태를 영상에 반영한다.
            return False

        def _apply_robot_eval_settings(self) -> None:
            head = self.robot_camera_names.get("head")
            self.env.apply_eval_settings(base_link_mass=E.EVAL_BASE_LINK_MASS,
                                         head_sensor=head.split("::")[1] if head else None,
                                         head_aperture=E.EVAL_HEAD_HORIZONTAL_APERTURE)

        def _load_instance_state(self, instance_id, instance_eval_state) -> None:
            pass  # 엔진은 판 묶음 전체를 _settle_and_finalize 에서 한 번에 불러온다

        def _settle_and_finalize(self, env_indices) -> None:
            self.env.load_instances({i: int(self.instance_eval_states[i].instance_id) for i in env_indices},
                                    mode=self.cfg.get("mode", "public_test"))

        # __exit__ 는 공식 그대로 (env.close + og.shutdown, 후자는 load_env 에서 할 일 없게 해 둠).
        # 덮지 않아야 tools/eval_instrumented.py 가 base 에 건 저장 훅(trace.npz)이 그대로 돈다.

    return PortedEvaluator


def main():
    ours, rest = _split_args(sys.argv[1:])
    from omnigibson.eval import eval as EV
    from omnigibson.eval import evaluator as E

    EV.BatchedEvaluator = make_ported_evaluator(E, ours.backend, ours.scene_root)
    if ours.instrument:
        # 원본 평가기 기록과 같은 도구로: eval_instrumented 가 공식 BatchedEvaluator(부모)에 훅을 걸고 EV.main() 을 부른다.
        # PortedEvaluator 는 그 부모를 상속하므로 같은 훅이 걸린다 -> 같은 키의 trace.npz (tools/trace_compare.py 로 비교).
        import importlib.util

        path = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "eval_instrumented.py"))
        spec = importlib.util.spec_from_file_location("eval_instrumented", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.argv = [path] + ours.instrument + ["--"] + rest
        mod.main()
        return
    sys.argv = ["omnigibson.eval.eval"] + rest
    EV.main()


if __name__ == "__main__":
    main()
