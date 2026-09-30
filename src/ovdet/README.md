# ovdet — open-vocabulary object detector (YOLOE, TensorRT FP16, C API)

`ovdet` is the detector of the scenemap perception stack (`docs/scenemap_설계.md`).

- **In:** one camera image and a prompt, the task's BDDL object names.
- **Out:** a list of objects. Each object has a prompt index, a score, a box and a mask.

The network is a YOLOE text-prompt segmentation head. It runs in TensorRT (FP16). Everything around the network is hand-written CUDA/C++:

- letterbox
- class selection under the prompt
- score sort
- greedy NMS
- mask assembly on the prototype grid
- area gate
- mask de-duplication

There is no Python and no ROS at run time, so the evaluator process calls it directly.

## API (`include/ovdet.h`)

```c
OvdConfig cfg; ovd_default_config(&cfg);
cfg.seg_engine = "yoloe-11l-all.plan"; cfg.names = "yoloe-11l-all.plan.names.txt";
OvdHandle* h = ovd_create(&cfg, err, sizeof err);
ovd_set_prompt(h, names, n, err, sizeof err);            // once per episode: the prompt table
const sm_detections* d = ovd_detect(h, &img, &timing);   // per image; valid until the next call
```

- **Output struct.** The output is `sm_detections`, the detector contract of `docs/scenemap_설계.md` 4.2, field for field. `scenemap.h` uses the same `SM_DETECTIONS_DEFINED` guard, so the two headers can be included together.
  - `cls`: index into the prompt table, i.e. the order of the names given to `ovd_set_prompt`.
  - `score`: the class score.
  - `box`: x0, y0, x1, y1 in input pixels.
  - `stamp` and `cam`: passed through from the input image.
- **Masks.** Masks stay on the detector's grid, which is the prototype grid: 256 x 256 for a 1024 input.
  - Each mask is a row-major bit array. Cell `k = j * mask_w + i` is bit `k & 31` (LSB first) of word `k >> 5`.
  - Input pixel = cell × `mask_s` + `mask_o`, which undoes the letterbox. Cell (i, j) covers x in [i·sx + ox, (i+1)·sx + ox).
  - Nothing is copied: the arrays belong to the handle.
- **Prompt names.** Names are matched to the engine's vocabulary after normalisation: `.n.NN` dropped, `_` → space, lower case. So `radio_receiver.n.01` matches `radio receiver`.
  - A name outside the vocabulary keeps its index, but is never detected.
  - Such names are reported in `err`.
- **Input image.** RGB, BGR or RGBA u8, in host or device memory, with any row stride.
- **Thread safety.** One handle per thread.

## Engines

The engine is built once with a whole vocabulary: every task's BDDL objects plus 18 scene structures, 272 names in `config/vocab_all.txt`.

A prompt switches classes on and off. YOLOE's class scores are independent sigmoids per class, so the result equals an engine exported with only the prompt's names. `config/task_prompts.txt` lists each task's names, plus the `_scene` line.

```
~/meridian_export_venv/bin/python tools/export_yoloe.py --model yoloe-11l-seg --vocab all --out ~/meridian_models/onnx/yoloe-11l-all.onnx   # CPU, Ultralytics
~/meridian_venv/bin/python tools/build_engines.py ~/meridian_models/onnx/yoloe-11l-all.onnx                                              # GPU lock
```

## Build (Linux / WSL; CUDA 12.8, TensorRT 10)

```
bash scripts/build_linux.sh        # -> ~/ovdet_build/libovdet.so, ovdet_smoke
```

The Windows host has no TensorRT SDK installed, so the library is built and run on Linux, which is also the submission Docker's OS.

## Detector comparison

`scripts/eval_linux.sh` runs `tools/ovdet_eval.py` on the same frames for every head:

- ep0, 0–40 s, every 5th frame
- ep200, whole episode, every 15th frame

Ground truth works as follows:

1. Depth pixels on a 2-px grid are placed in the map with the ground-truth camera pose.
2. Each point is labelled by `gt_scene` with the GT object whose box contains it.

The FastSAM + CLIP row runs the retired `deprecated/ovdet_fastsam` library.

The results and the reasoning behind the choice are in `docs/ovdet_검출기.md`.

## Licence (AGPL-3.0)

YOLOE's code and weights are AGPL-3.0:

- Ultralytics (`ultralytics` 8.4, `yoloe-11*-seg.pt`)
- THU-MIG (`THU-MIG/yoloe`)

The TensorRT engines are derived from those weights. ovdet itself contains no Ultralytics code.

The competition submission (Docker image given to the organizers) therefore ships AGPL-covered weights. On 2026-09-30 the user decided that **the submission's source is published under AGPL-3.0**. `docs/제출지침.md` lists this as a submission checklist item: a source link and the LICENSE go into the README.
