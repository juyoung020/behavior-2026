# src/sim/engine — our own simulator engine (training/dev only)

Goal: a GPU-batched simulator for BEHAVIOR tasks (thousands to millions of parallel episodes) for agent / RL / VLA
training. It never replaces the official evaluator (organizers evaluate with official OmniGibson / Isaac Sim), so every
layer is verified numerically against the official stack. Design, numbers and status: `docs/엔진_자체구현.md` (Korean).

PhysX 5.6.1 (tag `107.3-omni-and-physx-5.6.1`, the exact version inside Isaac Sim 5.1) is used **only as the oracle**.
The engine itself is hand-written C++ (then CUDA) and does not link PhysX.

| Path | What |
|---|---|
| `core/` | The engine (header-only, no PhysX). `pmath.h` math with PhysX-identical operation order; `rigid.h` free rigid bodies (TGS no-constraint path, sleep) |
| `tests/` | Layer-1 tests: engine vs PhysX oracle, bitwise. Only test binaries link PhysX |
| `replay/` | Layer-0 oracle: rebuild a recorded OmniPVD (`.ovd`) scene with our PhysX build and replay it bit-for-bit (`ovd_replay`), plus `ovd_dump`, `ovd_diff`, `ovd_selftest` |
| `capture/` | Record the official evaluator (OVD + convex hulls + omni filter tables + side log of non-OVD calls) without modifying it |
| `scripts/` | Build PhysX / tools on Linux, count per-task physics needs from BDDL |

Status (2026-09-29)
- Layer 0 self-test: record → replay of a BEHAVIOR-like scene (articulation drives, mimic joint, fixed joint attach/detach,
  aggregates, convex/trimesh, teleports, forces, sleep/wake, 300-body convex pile) is bit-identical for every step.
- Layer 1: hand-written free rigid bodies match PhysX bit-for-bit (3,000 bodies × 1,200 steps).
- Official evaluator recording: pending (GPU reserved for another task).

Build (Linux, Ubuntu 22.04, clang 14, CUDA 12.8), from the repo root. `build_replay.sh` still reads its sources from `/mnt/c/behavior-2026`, which on this PC is a link to the repo.
```
bash src/sim/engine/scripts/build_physx.sh linux-carbonite checked   # ~/engine-deps (not in git)
bash src/sim/engine/scripts/build_replay.sh checked                   # ~/engine-build/replay-checked
cmake -S src/sim/engine/tests -B ~/engine-build/tests -DCMAKE_CXX_COMPILER=clang++ && cmake --build ~/engine-build/tests
```

Never commit decrypted assets or anything extracted from scenes (`dumps/`, `*.ovd` are git-ignored).
