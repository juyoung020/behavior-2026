"""Launcher: official evaluator (omnigibson.eval.eval, unmodified) with --policy local, whose LocalPolicy gets the
native pi0.5 engine as its policy (LocalPolicy.forward -> self.policy.act(obs), eval/policies.py:28-35).

    python run_eval_native.py --weights <repo>/data/pi05_native/pi05_radio.pi05w [--prompt ...]
        [--replan 16] [--native-log out.csv] -- <omnigibson.eval.eval arguments, e.g. --task-name turning_on_radio>

The engine lives in this (the simulator's) process, so the policy's GPU memory is not "another process" for the
Isaac Sim renderer. Nothing under BEHAVIOR-1K/ is changed: the patch is applied to the class at runtime.
"""
import argparse
import json
import os
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / ("build_win" if os.name == "nt" else "build")))
sys.path.insert(0, str(HERE))


def task_prompt(task: str) -> str:
    reg = json.loads((HERE / "b1k_tasks.json").read_text(encoding="utf-8"))  # openpi TASK_REGISTRY["b1k"]
    if task in reg:
        return reg[task]
    meta = HERE.parents[2] / "data/2026-challenge-demos/meta/tasks.jsonl"
    for line in meta.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("task_name") == task:
            return d["task"]
    raise SystemExit(f"no prompt for task {task}; pass --prompt")


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--replan", type=int, default=16)
    ap.add_argument("--native-log", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--policy" in eval_args:
        i = eval_args.index("--policy")
        del eval_args[i:i + 2]
    task = eval_args[eval_args.index("--task-name") + 1]
    prompt = args.prompt or task_prompt(task)

    from pi05_policy import Pi05NativePolicy
    import omnigibson.eval.policies as P

    orig_init = P.LocalPolicy.__init__
    holder = {}

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.policy = Pi05NativePolicy(args.weights, prompt, replan_every=args.replan, log_path=args.native_log,
                                       seed=args.seed)
        holder["p"] = self.policy

    P.LocalPolicy.__init__ = init
    sys.argv = ["omnigibson.eval.eval", *eval_args, "--policy", "local"]
    try:
        runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)
    finally:
        if "p" in holder:
            holder["p"].flush()


if __name__ == "__main__":
    main()
