# behavior-2026 — BEHAVIOR Challenge 2026 workspace

**Idea: a dynamic 3D scene graph + an AI agent (or RL planner) + a VLA, combined.**

- The scene map (scenemap, built from scratch) keeps a 2D SLAM map and registers every object found by an open-vocabulary YOLOE detector at its 3D position. This is the robot's memory.
- The agent uses that graph for long-horizon planning, step tracking and failure recovery.
- The VLA (π0.5) turns the current step instruction plus the three cameras into actions.

Implementation rule: zero bottlenecks. Hot paths are hand-written native code (C++/CUDA, Rust for orchestration). No PyTorch in our execution paths.

- Plan and decision log: [plan.md](plan.md)
- Research notes, rules and run logs (Korean): [docs/](../../README.md)
- Deadline: 2026-10-16 AoE (KST 10-17 20:59)

## Repository layout

```
BEHAVIOR-1K/          challenge framework (StanfordVL, tag v3.9.3-post1) — submodule, never modified
  datasets/           simulator assets, task instances, decryption key (not in git)
data/                 2026 challenge demos (LeRobot v3): metadata + task 0 only (not in git)
src/
  agent/              high-level planner agent + evaluator↔policy relay (Rust, raw OpenAI-compatible API)
  pi05_native/        π0.5 inference engine, hand-written C++/CUDA (in progress)
  engine/             our own GPU simulator engine; layer 0 = PhysX 5.6.1 oracle replay (C++)
  fasteval/           evaluator acceleration: chunked-replay policy server, instrumentation
  fasttrain/          training data pipeline: NVDEC + fused CUDA kernels, Rust indexer
  scenemap/           2D SLAM + object map + planner queries (C++/CUDA, Rust)
  ovdet/              open-vocabulary detector (YOLOE, TensorRT, C API; AGPL-3.0)
  configs/            evaluator robot configs
tools/                run, measure and verify scripts (evaluator launcher, trace_compare, black-frame checks, …)
  setup/              one-time install/download scripts
refs/                 reference repos (2025 top teams) — submodules
docs/                 documentation (Korean); raw/ = verbatim copies of official pages
outputs/ logs/        evaluation results (JSON; videos are not in git) and logs
plan.md               plan and decisions
```

## Environments

**Windows (`C:\behavior-2026`)**
- conda env `behavior`: Python 3.11, Isaac Sim 5.1, OmniGibson (eval), BDDL, JoyLo.
- Official evaluator: `conda activate behavior` → `python -m omnigibson.eval.eval ...`, or use `tools/run_eval_radio.ps1`.

**WSL (`Ubuntu-22.04`, user `juyoung`)**

```
~/openpi                 π0.5 reference server (wensi-ai/openpi, behavior branch)
~/openpi-comet           2025 2nd-place code adapted to the 2026 evaluator (our changes: src/comet/patches, applied by tools/setup/setup_comet_wsl.sh)
~/checkpoints/           π0.5 radio, GR00T N1.7 radio, 2025 1st-place submission, Comet pt50
~/engine-deps/           PhysX 5.6.1 source + build (engine oracle)
~/.config/behavior-2026/ secrets (KAU API key), never committed — see .env.example
```

**Linux (`ad17-MS-7E01`, RTX 4090, Ubuntu 22.04.5)**
- Official evaluator on Linux: conda env `behavior` (Isaac Sim 5.1, torch 2.7.0+cu128, warp-lang 1.12.0), NVIDIA driver 580.178.04-open (535 failed — see [docs/Linux_설치.md](../../Linux_설치.md)).
- First check (`tools/setup/linux_first_check.sh`): 5 zero-action runs + 2 replays, 0 black frames — black-frame issue (B) does not occur on Linux, so submission runs use this PC.

## Setup notes

- Submodules: `git submodule update --init` (BEHAVIOR-1K, refs, Spark-DSG).
- Commits carry no Claude co-author lines: `.claude/settings.json` turns attribution off, and `tools/git-hooks/commit-msg` strips any that slip through. Run once per clone: `git config core.hooksPath tools/git-hooks`.
- Data, assets, keys, model weights and videos are not in git. Download scripts live in `tools/setup/`, and install notes in [docs/설치기록.md](설치기록.md).
- Known issue on this Windows PC: if another process holds more than about 5 GiB of GPU memory, Isaac Sim 5.1 returns all-black RGB to the policy (depth is fine). See plan.md §4.0.
  - Do not run the π0.5 server on the same GPU as the simulator.
  - `tools/black_frame_check.py` flags invalid runs.
