# pi05_train — native C++/CUDA training for openpi π0.5 (no JAX, no PyTorch)

Status: ported and verified; no real training run has been started (porting and verification only).

- **expert mode** (action expert + heads trainable, vision/language frozen): loss, all gradients and the AdamW/EMA
  updates after 1 and 10 steps match openpi's JAX training step within 2× of the JAX CPU-vs-GPU difference on a cut
  model (2 SigLIP + 2 Gemma layers of the real radio checkpoint). One Adam step-1 point is an exception: a sign flip
  on a gradient element that is almost zero.
- **lora mode** (openpi's pi05 LoRA variant: SigLIP fully trainable, Gemma rank-16 and expert rank-32 LoRA, heads):
  same check. 247 of 251 points pass: two LayerNorm-scale gradients are at 2.0× and 2.5× of the floor, and two
  step-1 Adam/EMA points differ by an ulp or a sign flip.
- **Randomness and augmentation**: openpi's key chain (`fold_in` → split → augmentation keys / normal noise / beta
  time) and augmax augmentation are reproduced. Keys are bit-identical, and the images match JAX at its own CPU/GPU
  level.
- **Driver**: `pi05_train` reads batches straight from the `src/fasttrain` native loader through its C API, with no
  Python in between. At full size (batch 32, optimizer state in pinned host memory) the expert mode needs 3.0 s per
  step and 10.7 GiB of GPU memory.

Korean write-up with the tables: [`docs/π05_네이티브엔진.md`](../../docs/π05_네이티브엔진.md), section 12.

## What is reproduced

| openpi (WSL `~/openpi`, `behavior` branch) | here |
|---|---|
| `models/pi0.py:189-214` `compute_loss` (flow matching, joint prefix+suffix pass) | `src/trainer.cu` (expert), `src/lora.cu` (lora); one sample at a time, gradients summed in f32 |
| `scripts/train.py:85-180` (frozen params in bf16, trainable f32 masters, `value_and_grad`, EMA 0.99) | `accumulate / finalize_grads / opt_step` |
| `training/optimizer.py` (clip_by_global_norm 1.0 → AdamW b1 .9 b2 .95 eps 1e-8 wd 1e-10, warmup-cosine lr) | `src/tparams.cu`, `src/tkern.cu adamw_step`; settings read from the state file |
| key chain `train.py` + `pi0.py:192-197`, `jax.random` split / uniform / normal / gamma / beta | `src/trng.h` |
| `models/model.py:168-187` augmax augmentation | `src/augment.cu` |
| `nn.remat(..., nothing_saveable)` per layer | lora mode keeps only layer inputs and recomputes each layer in backward |
| bf16 rounding of every JAX op, including the reverse-mode ops JAX generates and their order | commented per kernel |

## Layout

| path | role |
|---|---|
| `src/tgemm.cuh` | general bf16 tensor-core GEMM (any operand layout via `ldmatrix`/`ldmatrix.trans`): openpi's `[in, out]` parameters are used as stored for y = xW, dx = dyWᵀ, dW = xᵀdy |
| `src/tkern.cu`, `src/tkern2.cu` | forward pieces and VJPs: adaRMS / RMSNorm, LayerNorm (f32 params), RoPE, f32 and bf16 softmax, gated residual, gelu, f32 dense layers, SigLIP stem, flow loss; AdamW, EMA, global norm |
| `src/tparams.cu/.h` | trainable parameter set (optionally host-offloaded Adam moments / EMA, streamed per step), frozen bf16 weights |
| `src/trainer.cu/.h` | expert mode; the frozen prefix runs through the inference engine (`../pi05_native`) |
| `src/lora.cu/.h` | lora mode; SigLIP + joint Gemma pass + LoRA, all in this trainer |
| `src/trng.h`, `src/augment.cu` | openpi's random numbers and augmentation |
| `tools/train_main.cu` | `pi05_train`: training loop fed by the fasttrain loader; saves params, EMA, Adam state, step; resumes |
| `tools/train_ref.py`, `tools/train_verify.cpp` | JAX reference (cut model) and the comparison |
| `tools/aug_ref.py`, `tools/aug_test.cu` | randomness / augmentation reference and check |
| `tools/make_state.py`, `tools/state_to_orbax.py` | full-size initial state from an orbax checkpoint; trained state back to an openpi params checkpoint |
| `tools/train_bench.cpp`, `tools/tgemm_test.cu`, `tools/session.sh` | memory/time measurement, GEMM check, GPU-lock sessions |

## Build and check (Linux / WSL)

```bash
bash src/pi05_native/build_linux.sh            # inference engine library (frozen prefix in expert mode)
bash src/fasttrain/build.sh                    # loader library (for the driver)
bash src/pi05_train/build_linux.sh             # -> ~/pi05_train_build
PI05_TRAIN_MODE=expert bash src/pi05_train/tools/session.sh gemm ref_gpu ref_cpu verify
PI05_TRAIN_MODE=lora   bash src/pi05_train/tools/session.sh ref_gpu ref_cpu verify
bash src/pi05_train/tools/session.sh aug
```

## Train (not run yet)

```bash
python tools/make_state.py --ckpt <openpi checkpoint> --mode expert|lora --out state.pi05d   # WSL openpi venv
pi05_train --state state.pi05d --model pi05.pi05w --table <fasttrain table dir> --lut <lut_w720.bin> \
           --steps 30000 --offload 2 --save-every 1000 --out <dir>
python tools/state_to_orbax.py --state <dir>/state_stepN.pi05d --base <openpi checkpoint> --out <new ckpt> --ema
```
