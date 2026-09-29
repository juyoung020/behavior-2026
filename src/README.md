# src — our implementation

Everything we build lives here. The official `BEHAVIOR-1K/` checkout is never modified; we wrap it from here.

Language rule: C++/CUDA or Rust by default, and zero bottlenecks is the principle.
- Python only where it is unavoidable: thin glue inside the official Python processes (OmniGibson evaluator, openpi), and one-off offline export tools.
- No PyTorch in our execution paths.

| Folder | What it is | Language |
|---|---|---|
| `agent/` | High-level planner agent (tool calling, memory, summaries, step loop over the scene graph) and the evaluator↔policy relay | Rust |
| `pi05_native/` | π0.5 inference engine, verified layer by layer against the JAX reference | C++/CUDA |
| `engine/` | Our own GPU simulator engine. Layer 0 replays PhysX 5.6.1 bit-exactly as the oracle; `core/` is hand-written physics bit-matched to PhysX | C++ (→ CUDA) |
| `fasteval/` | Evaluator acceleration: chunked-replay policy server, instrumentation | C++ planned, Python glue today |
| `fasttrain/` | Training data pipeline: NVDEC decode + fused color/resize kernels, mp4 indexer | C++/CUDA, Rust |
| `meridian/` | Build, run and test scripts for `meridian_ws/` (the dynamic 3D scene graph) | shell, C++ |
| `configs/` | Evaluator robot configs | YAML |
