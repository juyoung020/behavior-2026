# pi05_native — pi0.5 inference engine in C++/CUDA

Hand-written inference for openpi `pi05` checkpoints (PaliGemma 3B = SigLIP So400m/14 + Gemma 2B, plus the
300M action expert with adaRMS, 10-step flow matching). No PyTorch, no JAX, no cuBLAS/cuDNN in the execution
path: tensor-core GEMMs (`mma.sync` bf16, cp.async pipeline), fused kernels for everything else, one CUDA graph for
the whole inference, a SentencePiece BPE tokenizer and the openpi input/output transforms in C++.

Verified layer by layer against the JAX model (JSBSim-style: dump every layer, compare, and also plant the
reference input of each layer to isolate one layer). The acceptance rule is "our error vs JAX-GPU is within 2x of
JAX-CPU vs JAX-GPU" at every point. Details and numbers (Korean): `docs/π05_네이티브엔진.md`.

## Layout

| Path | What |
|---|---|
| `src/gemm.cuh` | bf16 tensor-core GEMM (`C = A·Bᵀ`, both K-major), tile configs, in-kernel deterministic split-K, fused epilogues that reproduce JAX's bf16 rounding |
| `src/kernels.cu` | SigLIP stem (f32), LayerNorm, RMSNorm/adaRMS, RoPE + KV cache, softmax, flow update, time embedding |
| `src/model.cu` | weights/buffers, forward (SigLIP → Gemma prefix + KV cache → 10 denoising steps), CUDA graph, debug taps |
| `src/tokenizer.cpp` | SentencePiece BPE (protobuf parsed by hand), byte fallback, user-defined symbols |
| `src/host_io.cpp` | B1K state extraction, z-score normalization (float64), state discretization, prompt, unnormalize, delta→absolute |
| `src/image.cpp` | `resize_with_pad` bit-exact with PIL bilinear |
| `src/engine_api.cpp`, `include/pi05_native.h` | C API: `pi05_create / pi05_infer / pi05_act` (B1K receding horizon) |
| `glue/` | CPython extension (raw C API + buffer protocol), evaluator policy, launchers |
| `server/pi05_server.cpp` | Linux websocket policy server (openpi msgpack protocol), hand-written RFC 6455 + msgpack |
| `tools/` | offline Python: weight export (orbax → `.pi05w`), JAX reference dumps; C++: `pi05_verify`, GEMM test/bench, tokenizer test |

## Build

- Linux / WSL: `bash build_linux.sh [build_dir]` (CUDA 12.8+, g++). Produces `libpi05.a`, `pi05_verify`, `pi05_server`.
- Windows: `build_windows.bat` (MSVC 2022 + CUDA 12.8 nvcc from the conda env `pi05build`). Produces
  `build_win/_pi05native.pyd` for the evaluator's Python 3.11, `pi05_verify.exe`, `tok_test.exe`.

Only sm_120 (RTX 50xx) is compiled by default; set `PI05_ARCH` (Linux) for other GPUs.

## Use

```
# weights (once, WSL openpi venv)
tools/wsl_py.sh tools/export_weights.py --ckpt <ckpt dir> --asset turning_on_radio --out data/pi05_native/pi05_radio.pi05w
# check against JAX dumps
pi05_verify --weights W.pi05w --ref data/pi05_native/ref --tag gpu --floor cpu --floor-single cpu_planted --check
# evaluator, in-process
powershell -File glue/run_eval_native.ps1 [-MaxSteps 600] [-Video]
# submission-style server
pi05_server --weights W.pi05w --port 8000
```
