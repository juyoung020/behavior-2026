# src — our implementation

Everything we build lives here. The official `BEHAVIOR-1K/` checkout is never modified; we wrap it from here.

Language rule: C++/CUDA or Rust by default, and zero bottlenecks is the principle.
- Python only where it is unavoidable: thin glue inside the official Python processes (OmniGibson evaluator, openpi), and one-off offline export tools.
- No PyTorch in our execution paths.

Three layers, the same split as the team repo `robot-agent` (① memory → ② planning → ③ action), plus `sim/` for everything around them.

| Folder | What it is | Language |
|---|---|---|
| `scene_graph/scenemap/` | ① 2D SLAM + detected-object map queried in-process by the planner (C ABI), saved as Spark-DSG. Design: `docs/scenemap_설계.md` | C++/CUDA |
| `scene_graph/ovdet/` | ① Open-vocabulary object detector for scenemap (YOLOE, TensorRT FP16, C API) | C++/CUDA |
| `agent/planner/` | ② High-level planner agent (OpenAI-style tool calling, memory, summaries, step boundaries over the scenemap object map, swappable decider for RL) and the zero-copy evaluator↔π0.5 relay that injects the step instruction; the π0.5-side glue is `tools/serve_b1k_agent.py`. Design: `docs/에이전트_설계.md` | Rust |
| `vla/pi05_native/` | ③ π0.5 inference engine, verified layer by layer against the JAX reference | C++/CUDA |
| `vla/pi05_train/` | ③ π0.5 training step | C++/CUDA |
| `vla/fasttrain/` | ③ Training data pipeline: NVDEC decode + fused color/resize kernels, mp4 indexer | C++/CUDA, Rust |
| `sim/engine/` | Our own GPU simulator engine. Layer 0 replays PhysX 5.6.1 bit-exactly as the oracle; `core/` is hand-written physics bit-matched to PhysX | C++ (→ CUDA) |
| `sim/fasteval/` | Evaluator acceleration: chunked-replay policy server, instrumentation | C++ planned, Python glue today |
| `sim/integ/` | Integration: `simlink` (planner + scenemap in one process) | Rust, shell |
| `sim/configs/` | Evaluator robot configs | YAML |

Modules the current pipeline no longer uses (instruction-format probe, Comet patches, Windows-only scripts, …) are kept under [`archive/`](../archive/README.md).
