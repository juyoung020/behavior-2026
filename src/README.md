# src — our implementation

Everything we build lives here. The official `BEHAVIOR-1K/` checkout is never modified; we wrap it from here.

Language rule: C++/CUDA or Rust by default, and zero bottlenecks is the principle.
- Python only where it is unavoidable: thin glue inside the official Python processes (OmniGibson evaluator), and one-off offline export tools.
- No PyTorch in our execution paths.

Three layers, the same split as the team repo `robot-agent` (① memory → ② planning → ③ action), plus `sim/` for everything around them.
③ (VLA) is not here: the π0.5 engine, trainer and data pipeline (`vla/`) were removed 10-06; VLA = RecallVLA (robot-agent `training/vla`).

| Folder | What it is | Language |
|---|---|---|
| `scene_graph/scenemap/` | ① 2D SLAM + detected-object map queried in-process by the planner (C ABI), saved as Spark-DSG. Design: `docs/scenemap_설계.md` | C++ |
| `scene_graph/ovdet/` | ① Detector for scenemap (TensorRT FP16, C API). Decided: ObjectSAM (YOLO26n student distilled from FastSAM-s, things-only, `yolo26n-seg-obj-416`) + SigLIP 2 B/32 (`scene_graph/clip/`) + scenemap objprob. Archived YOLOE/YOLO-seg engines stay selectable | C++/CUDA |
| `scene_graph/clip/` | ① sgclip: object crop → SigLIP 2 B/32 image embedding (TensorRT), label-table lookup, vectors (original) and name cache (`cache/`) in the memory folder. In progress | C++/CUDA |
| `scene_graph/runtime/` | ① sgrt: one C ABI that runs object memory inside the evaluator/robot process (scenemap + detector, periodic save). Pose source `SGRT_POSE=slam\|odom\|gt` | C++/CUDA, Python glue |
| `scene_graph/spark_dsg/` | ① Our copy of Spark-DSG (MIT-SPARK, v1.1.3, BSD-3), cut down to objects + rooms (places are computed in the backend); scenemap builds it first. Changes: `OUR_CHANGES.md` | C++ |
| `scene_graph/da/` | ① Data association: merges duplicate confirmed objects (same label, overlapping boxes) left by per-frame segments into one object; compiled into scenemap | C++ |
| `scene_graph/sgview/` | ① **The** scene-graph viewer (Spark-DSG 장면 그래프 보기 = sgview). Live object-memory viewer in the browser: Rust server (memory folder, or a live socket stream from sgrt relayed over SSE) + three.js, no Python / Spark-DSG | Rust, C++ (walls), JS |
| `scene_graph/viewer/` | LEGACY sgviz: the old Python viewer (Spark-DSG + viser). Polls files, not real-time. Do not use — use `sgview` (robot-agent `tools/run_sgview.sh`, `tools/run_explore_live.sh`). Kept for history | Python (offline, legacy) |
| `agent/planner/` | ② High-level planner agent (OpenAI-style tool calling, memory, summaries, step boundaries over the scenemap object map, swappable decider for RL) and the zero-copy evaluator↔VLA relay that injects the step instruction. Design: `docs/에이전트_설계.md` | Rust |
| `sim/engine/` | Our own GPU simulator engine. Layer 0 replays PhysX 5.6.1 bit-exactly as the oracle; `core/` is hand-written physics bit-matched to PhysX | C++ (→ CUDA) |
| `sim/fasteval/` | Evaluator acceleration: replay policy server, trace compare, instrumentation | C++ planned; Rust (`replaysrv`, `tracecmp`, `npz`) and Python glue today |
| `sim/integ/` | Integration: `simlink` (planner + scenemap in one process) | Rust, shell |
| `sim/explore/` | One simulator run of robot-agent's explore skill (evaluator side), 8080 sgview launcher (`viewer_8080.sh`) | Python, shell |
| `sim/move_robot/` | Simulator side of the `move_robot` tool; calls robot-agent's Rust crate `src/agent/tools/move_robot` via ctypes | Python |
| `sim/configs/` | Evaluator robot configs | YAML |

Modules the current pipeline no longer uses (instruction-format probe, Comet patches, Windows-only scripts, …) are kept under [`archive/`](../archive/README.md).
