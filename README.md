![scene graph viewer: objects / places / rooms, robot GT trajectory](docs/img/cover.png)

# behavior-2026 — BEHAVIOR Challenge 2026 workspace

**Idea: a dynamic 2D scene graph + an AI agent (or RL planner) + a VLA, combined.**

- The scene map (scenemap, built from scratch) keeps a 2D SLAM map and registers every detected object at an xyz position (today the open-vocabulary YOLOE detector `ovdet`; moving to FastSAM-s masks + SigLIP 2 embeddings, `scene_graph/clip`): the segmentation mask gives the object's centroid, and camera depth turns it into xyz. Objects are shown on that 2D map (the viewer is 2D). This is the robot's memory.
- The agent uses that graph for long-horizon planning, step tracking and failure recovery.
- The VLA (π0.5) turns the current step instruction plus the three cameras into actions.

Implementation rule: zero bottlenecks. Hot paths are hand-written native code (C++/CUDA, Rust for orchestration). No PyTorch in our execution paths.

- Plan and decision log: [plan.md](plan.md)
- Research notes, rules and run logs (Korean): [docs/](docs/README.md)
- Deadline: 2026-10-16 AoE (KST 10-17 20:59)

## Documents

- [docs/README.md](docs/README.md) — index of every design note and run log (Korean)
- [docs/scenemap_설계.md](docs/scenemap_설계.md) — object memory design: 2D SLAM, object map, planner queries, pose source
- [archive/README.md](archive/README.md) — modules the current pipeline no longer uses: what, why, how to revive
- [tools/README.md](tools/README.md) — Linux runners, old Windows/WSL → Linux table
- [src/scene_graph/spark_dsg/OUR_CHANGES.md](src/scene_graph/spark_dsg/OUR_CHANGES.md) — what we changed in our Spark-DSG copy
- Team repo (robot-agent): [docs/clip_candidates.md](https://github.com/juyoung020/robot-agent/blob/main/docs/clip_candidates.md) (image–text embedding candidates), [training/README.md](https://github.com/juyoung020/robot-agent/blob/main/training/README.md) (training the small models that `scene_graph/clip` runs)

## Decisions

- Image embedding: SigLIP 2 B/32.
- Segmentation: FastSAM-s at 416.
- Object vectors are kept as the original embeddings; names are derived and cached in the memory folder's `cache/`.
- CUDA 12.8 (`/usr/local/cuda-12.8`) is the build and bit-verification baseline; 13.2 is installed but not used.
- Map pose comes from `SGRT_POSE=slam|odom|gt` (real robot default `slam`; simulator tests use `gt`, map = world).

## Repository layout

```
BEHAVIOR-1K/          challenge framework (StanfordVL, tag v3.9.3-post1) — submodule, never modified
  datasets/           simulator assets, task instances, decryption key (not in git)
data/                 2026 challenge demos (LeRobot v3): metadata + task 0 only (not in git)
src/                  three layers, same as the team repo (robot-agent): ① memory → ② planning → ③ action, plus sim/
  scene_graph/        ① object memory
    scenemap/         2D SLAM + object map + planner queries, Spark-DSG save (C++/CUDA, Rust)
    ovdet/            open-vocabulary detector (YOLOE, TensorRT, C API; AGPL-3.0)
    clip/             sgclip: object crop → SigLIP 2 image embedding (TensorRT), label table lookup, vectors and name cache in the memory folder (C++/CUDA; in progress)
    runtime/          sgrt: one C ABI that runs object memory inside the evaluator/robot process (scenemap + ovdet, periodic save) (C++/CUDA)
    spark_dsg/        our copy of Spark-DSG (MIT-SPARK, v1.1.3, BSD-3), cut down to objects + rooms; scenemap builds it first. Changes: OUR_CHANGES.md
    da/               data association: merges the per-frame segments of one object into a single object (C++, built into scenemap)
    sgview/           live memory viewer in the browser (Rust server + three.js, no Python / Spark-DSG)
    viewer/           sgviz: the old Python viewer (Spark-DSG + viser), to be removed
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
    explore/          one simulator run of the explore skill (evaluator side) + 8080 viewer launcher
    move_robot/       simulator side of the move_robot tool (calls robot-agent's Rust crate via ctypes)
    configs/          evaluator robot configs
tools/                run, measure and verify scripts (evaluator launcher, trace_compare, black-frame checks, …) (tools/README.md: Linux runners, old Windows/WSL → Linux table)
  setup/              one-time install/download scripts
  git-hooks/          commit-msg hook (strips Claude co-author lines)
archive/              modules the current pipeline no longer uses, kept as they were (archive/README.md: what, why, how to revive)
refs/                 reference repos (2025 top teams) — submodules
docs/                 documentation (Korean); raw/ = verbatim copies of official pages
outputs/ logs/        evaluation results (JSON; videos are not in git) and logs
plan.md               plan and decisions
```

## Environments

**Linux PC (`jy-desktop`, `~/robot-agent/src/behavior-2026`)** — the work machine
- Ubuntu 22.04, RTX 5070 Ti 16 GB (sm_120), NVIDIA driver 580.178.04-open (595 segfaults Isaac Sim 5.1), CUDA 12.8 (see Decisions). Details: [docs/Linux_설치.md](docs/Linux_설치.md).
- conda env `behavior`: Isaac Sim 5.1, OmniGibson 3.9.3 (eval), warp-lang 1.12.0.
- Official evaluator: `conda activate behavior` → `python -m omnigibson.eval.eval ...`.
- Windows/WSL scripts were replaced by Linux ones: mapping table in [tools/README.md](tools/README.md).

**Earlier PC (`ad17-MS-7E01`, RTX 4090 24 GB, 10-02)** — record only
- First check (`tools/setup/linux_first_check.sh`): 5 zero-action runs + 2 replays, 0 black frames.
- At that time `~/openpi` and secrets in `~/.config/behavior-2026/` (see .env.example) were not set up yet.

## Setup notes

- Submodules: `git submodule update --init` (BEHAVIOR-1K, refs). Spark-DSG is no longer a submodule; it is vendored in `src/scene_graph/spark_dsg/`.
- Commits carry no Claude co-author lines: `.claude/settings.json` turns attribution off, and `tools/git-hooks/commit-msg` strips any that slip through. Run once per clone: `git config core.hooksPath tools/git-hooks`.
- Data, assets, keys, model weights and videos are not in git. Download scripts live in `tools/setup/`, and install notes in [docs/Linux_설치.md](docs/Linux_설치.md) (Windows-era notes: [docs/archive/windows/README_최상위.md](docs/archive/windows/README_최상위.md)).
