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
- Windows (no longer used; Linux is the only work machine): `archive/src/vla/pi05_native/build_windows.bat`.

Only sm_120 (RTX 50xx) is compiled by default; set `PI05_ARCH` (Linux) for other GPUs.

## Use

```
# weights (once, WSL openpi venv)
tools/wsl_py.sh tools/export_weights.py --ckpt <ckpt dir> --asset turning_on_radio --out data/pi05_native/pi05_radio.pi05w
# check against JAX dumps
pi05_verify --weights W.pi05w --ref data/pi05_native/ref --tag gpu --floor cpu --floor-single cpu_planted --check
# evaluator, in-process
python glue/run_eval_native.py --weights data/pi05_native/pi05_radio.pi05w -- --task-name turning_on_radio ...
# (Windows launcher run_eval_native.ps1 moved to archive/)
# submission-style server
pi05_server --weights W.pi05w --port 8000
```

## Batched inference (`include/pi05_batch.h`)

`pi05_create_batch(weights, device, max_batch)` plus:

- `pi05_infer_batch`: one inference per episode for n episodes. Each episode's prefix runs through the
  single-inference CUDA graph. Then the denoising steps of all episodes run as one batched suffix.
- `pi05_act_batch`: the B1K wrappers for n environment slots, run natively (openpi receding horizon; the 2025 1st
  place wrapper with 26→20 cubic resampling, 4 kept actions for soft inpainting, stage voting and correction rules).
  It returns f32 `[n][23]` actions, the wrappers' action dtype.

Images are taken as the evaluator sends them: RGB or RGBA, any size, strided, host or device. Views that are not
224x224 go through openpi `resize_with_pad` (PIL bilinear), reproduced exactly; device views are resized on the GPU.

Every episode's actions are bit-identical to the single-episode API at every batch size (`tools/batch_test.cpp`).
Throughput on an RTX 5070 Ti:

| model | n = 1 | n ≥ 16 |
|---|---|---|
| radio | 86 ms | ~71 ms per episode, 14 episodes/s |
| 2025 1st place | 105 ms | ~75 ms per episode, 13.4 episodes/s |

The prefix (SigLIP + Gemma 2B, about 60 ms per episode) is compute-bound at about 90% of the GEMM ceiling, so
batching the prefix too does not help.

## Submission server (`server/`)

`pi05_server` speaks the BEHAVIOR / openpi websocket protocol: metadata on connect, msgpack observations
(`__ndarray__` maps), `{"action", "server_timing"}` replies, `{"reset": true}`, `__action_chunk_size__`, and
`GET /healthz`.

- All environments of a request go through one `pi05_act_batch` call.
- With `--task-map task_checkpoint_mapping.json --weights-dir DIR` (2025 1st place), the observation's `task_id`
  selects the checkpoint. Up to `--max-engines` checkpoints (default 3, about 20 GB) stay resident, evicted least
  recently used first.

The official client is `WebsocketClientPolicy` from BEHAVIOR-1K. Its actions are bit-identical to running the same
observations in process (`tools/server_check.sh`: radio and 1st place, 2 envs × 60 steps, plus a chunk-request run).

Image (`server/Dockerfile`, CUDA 12.8 base, no Python inside):

```bash
# build context: src/vla/pi05_native/ and weights/ (pb2025_ckpt1..4.pi05w + task_checkpoint_mapping.json, or pi05_radio.pi05w)
docker build -f src/vla/pi05_native/server/Dockerfile -t behavior-policy:pb2025 --build-arg MODEL=pb2025 .
docker run --gpus all -p 8000:8000 behavior-policy:pb2025
python -m omnigibson.eval.evaluator ... --policy websocket --host <server> --port 8000   # official evaluator
```

GPUs: sm_75 (Turing, e.g. TitanRTX) and newer; the image carries sm_75 / 80 / 86 / 89 / 90 / 120 code and the driver
picks the matching one. Turing has no bf16 tensor cores, so its GEMMs convert bf16 tiles to fp16 in shared memory
and multiply on fp16 tensor cores with fp32 accumulation. This is exact except for values outside the fp16 range.
Forced on an RTX 5070 Ti (`-DPI05_FP16_MMA`), its final actions stay inside the JAX CPU-vs-GPU difference for radio and
the 1st place.
