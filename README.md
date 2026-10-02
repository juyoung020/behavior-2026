# behavior-2026 — BEHAVIOR Challenge 2026 workspace

**Idea: a dynamic 2D scene graph + an AI agent (or RL planner) + a VLA, combined.**

- The scene map (scenemap, built from scratch) keeps a 2D SLAM map and registers every object found by an open-vocabulary YOLOE detector at an xyz position: the detector's segmentation mask gives the object's centroid, and camera depth turns it into xyz. Objects are shown on that 2D map (the viewer is 2D). This is the robot's memory.
- The agent uses that graph for long-horizon planning, step tracking and failure recovery.
- The VLA (π0.5) turns the current step instruction plus the three cameras into actions.

Implementation rule: zero bottlenecks. Hot paths are hand-written native code (C++/CUDA, Rust for orchestration). No PyTorch in our execution paths.

- Plan and decision log: [plan.md](plan.md)
- Research notes, rules and run logs (Korean): [docs/](docs/README.md)
- Deadline: 2026-10-16 AoE (KST 10-17 20:59)

## Repository layout

```
BEHAVIOR-1K/          challenge framework (StanfordVL, tag v3.9.3-post1) — submodule, never modified
  datasets/           simulator assets, task instances, decryption key (not in git)
data/                 2026 challenge demos (LeRobot v3): metadata + task 0 only (not in git)
src/                  three layers, same as the team repo (robot-agent): ① memory → ② planning → ③ action, plus sim/
  scene_graph/        ① object memory
    scenemap/         2D SLAM + object map + planner queries, Spark-DSG save (C++/CUDA, Rust)
    ovdet/            open-vocabulary detector (YOLOE, TensorRT, C API; AGPL-3.0)
  agent/              ② high-level planning
    planner/          planner agent + evaluator↔policy relay (Rust, raw OpenAI-compatible API)
  vla/                ③ low-level action (π0.5)
    pi05_native/      π0.5 inference engine, hand-written C++/CUDA
    pi05_train/       π0.5 training step in C++/CUDA
    fasttrain/        training data pipeline: NVDEC + fused CUDA kernels, Rust indexer
  sim/                simulator, evaluation and integration
    engine/           our own GPU simulator engine; layer 0 = PhysX 5.6.1 oracle replay (C++)
    fasteval/         evaluator acceleration: chunked-replay policy server, instrumentation
    integ/            evaluator ↔ planner ↔ scenemap link (simlink, Rust)
    configs/          evaluator robot configs
tools/                run, measure and verify scripts (evaluator launcher, trace_compare, black-frame checks, …)
archive/              modules the current pipeline no longer uses, kept as they were (archive/README.md: what, why, how to revive)
  setup/              one-time install/download scripts
refs/                 reference repos (2025 top teams) — submodules
docs/                 documentation (Korean); raw/ = verbatim copies of official pages
outputs/ logs/        evaluation results (JSON; videos are not in git) and logs
plan.md               plan and decisions
```

## Environments

**Linux PC (`ad17-MS-7E01`, `~/behavior-2026`)** — the only work machine
- Ubuntu 22.04.5, RTX 4090 24 GB (sm_89), NVIDIA driver 580.178.04-open (535 failed), CUDA toolkit 11.8 only so far. Details: [docs/Linux_설치.md](docs/Linux_설치.md).
- conda env `behavior`: Isaac Sim 5.1, OmniGibson (eval), torch 2.7.0+cu128, warp-lang 1.12.0.
- Official evaluator: `conda activate behavior` → `python -m omnigibson.eval.eval ...`.
- First check (`tools/setup/linux_first_check.sh`): 5 zero-action runs + 2 replays, 0 black frames.
- Not here yet: `~/openpi`, secrets in `~/.config/behavior-2026/` (see .env.example). The π0.5 radio checkpoint is being downloaded to `~/checkpoints/zips/pi05_turning_on_radio`.

## Setup notes

- Submodules: `git submodule update --init` (BEHAVIOR-1K, refs, Spark-DSG).
- Commits carry no Claude co-author lines: `.claude/settings.json` turns attribution off, and `tools/git-hooks/commit-msg` strips any that slip through. Run once per clone: `git config core.hooksPath tools/git-hooks`.
- Data, assets, keys, model weights and videos are not in git. Download scripts live in `tools/setup/`, and install notes in [docs/Linux_설치.md](docs/Linux_설치.md) (Windows-era notes: [docs/archive/windows/README_최상위.md](docs/archive/windows/README_최상위.md)).
