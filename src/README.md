# src — our implementation

Everything we build lives here. The official `BEHAVIOR-1K/` checkout is never modified; we wrap it from here.

Language rule: C++/CUDA or Rust by default, and zero bottlenecks is the principle.
- Python only where it is unavoidable: thin glue inside the official Python processes (OmniGibson evaluator, openpi), and one-off offline export tools.
- No PyTorch in our execution paths.

| Folder | What it is | Language |
|---|---|---|
| `agent/` | High-level planner agent (OpenAI-style tool calling, memory, summaries, step boundaries over the scenemap object map, swappable decider for RL) and the zero-copy evaluator↔π0.5 relay that injects the step instruction; the π0.5-side glue is `tools/serve_b1k_agent.py`. Design: `docs/에이전트_설계.md` | Rust |
| `pi05_native/` | π0.5 inference engine, verified layer by layer against the JAX reference | C++/CUDA |
| `engine/` | Our own GPU simulator engine. Layer 0 replays PhysX 5.6.1 bit-exactly as the oracle; `core/` is hand-written physics bit-matched to PhysX | C++ (→ CUDA) |
| `fasteval/` | Evaluator acceleration: chunked-replay policy server, instrumentation | C++ planned, Python glue today |
| `fasttrain/` | Training data pipeline: NVDEC decode + fused color/resize kernels, mp4 indexer | C++/CUDA, Rust |
| `scenemap/` | 2D SLAM + detected-object map queried in-process by the planner (C ABI). Design: `docs/scenemap_설계.md` | C++/CUDA |
| `ovdet/` | Open-vocabulary object detector for scenemap (YOLOE, TensorRT FP16, C API) | C++/CUDA |
| `integ/` | Integration: `simlink` (planner + scenemap in one process) | Rust, shell |
| `configs/` | Evaluator robot configs | YAML |
