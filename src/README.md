# src — our implementation

Everything we build lives here. The official `BEHAVIOR-1K/` checkout is never modified; we wrap it from here.

Language rule: C++/CUDA or Rust by default, and zero bottlenecks is the principle.
- Python only where it is unavoidable: thin glue inside the official Python processes (OmniGibson evaluator, openpi), and one-off offline export tools.
- No PyTorch in our execution paths.

Three layers, the same split as the team repo `robot-agent` (① memory → ② planning → ③ action), plus `sim/` for everything around them.

| Folder | What it is | Language |
|---|---|---|
| `scene_graph/scenemap/` | ① 2D SLAM + detected-object map queried in-process by the planner (C ABI), saved as Spark-DSG. Design: `docs/scenemap_설계.md` | C++ |
| `scene_graph/ovdet/` | ① Open-vocabulary object detector for scenemap (YOLOE, TensorRT FP16, C API). Object recognition is moving to FastSAM-s 416 + SigLIP 2 B/32 in `scene_graph/clip/` (in progress) | C++/CUDA |
| `scene_graph/clip/` | ① sgclip: object crop → SigLIP 2 B/32 image embedding (TensorRT), label-table lookup, vectors (original) and name cache (`cache/`) in the memory folder. In progress | C++/CUDA |
| `scene_graph/runtime/` | ① sgrt: one C ABI that runs object memory inside the evaluator/robot process (scenemap + detector, periodic save). Pose source `SGRT_POSE=slam\|odom\|gt` | C++/CUDA, Python glue |
| `scene_graph/spark_dsg/` | ① Our copy of Spark-DSG (MIT-SPARK, v1.1.3, BSD-3), cut down to objects + rooms (places are computed in the backend); scenemap builds it first. Changes: `OUR_CHANGES.md` | C++ |
| `scene_graph/da/` | ① Data association: merges duplicate confirmed objects (same label, overlapping boxes) left by per-frame segments into one object; compiled into scenemap | C++ |
| `scene_graph/sgview/` | ① Live object-memory viewer in the browser: Rust server (memory folder, or a live socket stream from sgrt relayed over SSE) + three.js, no Python / Spark-DSG | Rust, C++ (walls), JS |
| `scene_graph/viewer/` | ① sgviz: the old Python live object-memory viewer in the browser (Spark-DSG + viser) | Python |
| `agent/planner/` | ② High-level planner agent (OpenAI-style tool calling, memory, summaries, step boundaries over the scenemap object map, swappable decider for RL) and the zero-copy evaluator↔π0.5 relay that injects the step instruction; the π0.5-side glue is `tools/serve_b1k_agent.py`. Design: `docs/에이전트_설계.md` | Rust |
| `vla/pi05_native/` | ③ π0.5 inference engine, verified layer by layer against the JAX reference | C++/CUDA |
| `vla/pi05_train/` | ③ π0.5 training step | C++/CUDA |
| `vla/fasttrain/` | ③ Training data pipeline: NVDEC decode + fused color/resize kernels, mp4 indexer | C++/CUDA, Rust |
| `sim/engine/` | Our own GPU simulator engine. Layer 0 replays PhysX 5.6.1 bit-exactly as the oracle; `core/` is hand-written physics bit-matched to PhysX | C++ (→ CUDA) |
| `sim/fasteval/` | Evaluator acceleration: chunked-replay policy server, instrumentation | C++ planned; Rust (`replaysrv`, `tracecmp`, `npz`) and Python glue today |
| `sim/integ/` | Integration: `simlink` (planner + scenemap in one process) | Rust, shell |
| `sim/explore/` | One simulator run of robot-agent's explore skill (evaluator side), 8080 viewer launcher | Python, shell |
| `sim/move_robot/` | Simulator side of the `move_robot` tool; calls robot-agent's Rust crate `src/agent/tools/move_robot` via ctypes | Python |
| `sim/configs/` | Evaluator robot configs | YAML |

Modules the current pipeline no longer uses (instruction-format probe, Comet patches, Windows-only scripts, …) are kept under [`archive/`](../archive/README.md).
