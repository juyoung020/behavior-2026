"""평가기 프로세스 안 정책 공장 -- tools\\eval_instrumented.py --native-policy=native_policy:pi05 가 부른다.

네이티브 π0.5 엔진(src\\pi05_native, C++/CUDA, `_pi05native`)을 공식 LocalPolicy 의 policy 로 넣는다.
src\\pi05_native\\glue\\run_eval_native.py 와 같은 연결이다(LocalPolicy.forward -> self.policy.act(obs), eval/policies.py:28-35).
평가기 코드(BEHAVIOR-1K)는 그대로고, 정책은 우리 제출물이다.

설정은 환경변수로 받는다(tools\\exp_run.ps1 의 native 설정이 판마다 넣는다):
    PI05_NATIVE_WEIGHTS  가중치(.pi05w)              기본 C:/behavior-2026/data/pi05_native/pi05_radio.pi05w
    PI05_NATIVE_PROMPT   지시문(없으면 과제 이름으로 run_eval_native.task_prompt)
    PI05_NATIVE_REPLAN   재계획 간격(기본 16)        PI05_NATIVE_SEED 난수 씨앗(기본 0)
    PI05_NATIVE_LOG      엔진 스텝 기록 CSV(없으면 안 씀)
"""
from __future__ import annotations

import os
import pathlib
import sys

_NATIVE = pathlib.Path(__file__).resolve().parents[2] / "vla" / "pi05_native"


def pi05(cfg):
    for p in (_NATIVE / ("build_win" if os.name == "nt" else "build"), _NATIVE / "glue"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from omnigibson.eval.policies import LocalPolicy
    from pi05_policy import Pi05NativePolicy
    from run_eval_native import task_prompt

    task = cfg.task.name
    policy = LocalPolicy(action_dim=None)
    policy.policy = Pi05NativePolicy(
        os.environ.get("PI05_NATIVE_WEIGHTS", "C:/behavior-2026/data/pi05_native/pi05_radio.pi05w"),
        os.environ.get("PI05_NATIVE_PROMPT") or task_prompt(task),
        replan_every=int(os.environ.get("PI05_NATIVE_REPLAN", "16")),
        log_path=os.environ.get("PI05_NATIVE_LOG") or None,
        seed=int(os.environ.get("PI05_NATIVE_SEED", "0")),
    )
    return policy
