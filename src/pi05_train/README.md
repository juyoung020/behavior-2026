# pi05_train — native C++/CUDA training for openpi π0.5 (no JAX, no PyTorch)

Status: **work in progress.** The "expert" mode (action expert + heads trainable, vision/language frozen) matches openpi's
JAX training step on a cut model (2 SigLIP + 2 Gemma layers of the real radio checkpoint). Loss, every gradient and
the AdamW/EMA parameter updates after 1 and 10 steps are within 2× of the JAX CPU-vs-GPU difference. One Adam step-1
point (`time_mlp_in/bias`) is outside that bound; see the Korean write-up. LoRA mode and the data-loader hookup are
being ported. No training run has been started (porting and verification only).

Korean write-up: [`docs/π05_네이티브엔진.md`](../../docs/π05_네이티브엔진.md) (training section).

## What is reproduced

| openpi (WSL `~/openpi`, `behavior` branch) | here |
|---|---|
| `models/pi0.py:189-214` `compute_loss` (flow matching, joint prefix+suffix pass) | `src/trainer.cu` forward/backward, one sample at a time, gradients summed in f32 |
| `scripts/train.py:85-180` (frozen params in bf16, trainable f32 masters, `value_and_grad`, EMA 0.99) | `Trainer::accumulate / finalize_grads / opt_step` |
| `training/optimizer.py` (clip_by_global_norm 1.0 → AdamW b1 .9 b2 .95 eps 1e-8 wd 1e-10, warmup-cosine lr) | `src/tkern.cu` `adamw_step`, `OptCfg::lr` (settings read from the reference file / training config) |
| bf16 rounding of every JAX op, including the reverse-mode ops JAX generates | commented per kernel in `src/tkern.cu` |

## Layout

| path | role |
|---|---|
| `src/tgemm.cuh` | general bf16 tensor-core GEMM (any operand layout via `ldmatrix`/`ldmatrix.trans`), so openpi's `[in, out]` parameters are used as stored for y = xW, dx = dyWᵀ and dW = xᵀdy |
| `src/tkern.cu/.cuh` | adaRMS, RoPE, softmax, gated residual, gelu·up, f32 dense layers, flow loss — forward pieces and their VJPs; AdamW, EMA, global norm |
| `src/trainer.cu/.h` | expert-mode trainer; the frozen prefix runs through the inference engine (`../pi05_native`) |
| `tools/train_ref.py` | JAX reference: builds the (cut) openpi model, dumps params, batch, loss, gradients, optimizer steps |
| `tools/train_verify.cpp` | compares the native trainer against the reference (floor = JAX on the other platform) |
| `tools/tgemm_test.cu` | GEMM check for all operand layouts |
| `tools/session.sh` | one GPU-lock session (tests, reference dumps, verification) |

## Build and check (Linux / WSL)

```bash
bash src/pi05_native/build_linux.sh            # inference engine library (frozen prefix)
bash src/pi05_train/build_linux.sh             # -> ~/pi05_train_build
bash src/pi05_train/tools/session.sh gemm ref_gpu ref_cpu verify
```
