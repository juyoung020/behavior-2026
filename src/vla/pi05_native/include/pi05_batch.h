/* Batched inference of the native pi0.5 engine (many environments per call).
 *
 * pi05_infer_batch: one model inference per episode for n episodes. Each episode's prefix (SigLIP + Gemma 2B) runs
 * through the same CUDA graph as a single inference; then the denoising steps of all n episodes run as one batched
 * suffix (the action expert's weights are read once per step for everybody).
 * pi05_act_batch: the B1K policy wrappers for n environment slots (openpi receding horizon for pi05; the 2025 1st
 * place wrapper for PiBehavior: 26 of 30 actions in 20 steps with cubic resampling, 4 kept for soft inpainting,
 * stage voting, correction rules), with the inferences the slots need this step done in one pi05_infer_batch.
 *
 * Bit identity: episode i of pi05_infer_batch equals pi05_infer / pi05_infer_pb with the same inputs, and
 * pi05_act_batch equals calling pi05_act for i = 0 .. n-1 in that order (same random stream order), for every n.
 * The batched kernels keep each episode's arithmetic exactly (same GEMM k order and split-K partition per episode,
 * same softmax / norm reductions).
 *
 * Images: what the evaluator hands the policy — H x W uint8 RGB or RGBA (alpha ignored, like obs[..., :3]) with a
 * row stride, on the host or on the device. Views that are not 224x224 go through openpi's resize_with_pad (PIL
 * bilinear, reproduced exactly; on the GPU for device views).
 */
#ifndef PI05_BATCH_H
#define PI05_BATCH_H
#include "pi05_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const uint8_t* data;    /* first pixel */
  int32_t h, w;
  int64_t row_stride;     /* bytes between rows */
  int32_t pix_stride;     /* bytes between pixels: 3 = RGB, 4 = RGBA */
  int32_t on_device;      /* 1: CUDA device memory on the engine's device */
} Pi05View;

typedef struct {
  int32_t n;                     /* episodes, 1 .. max_batch */
  const Pi05View* views;         /* [n][3]: head, left wrist, right wrist (Pi05Info.cam_keys order) */
  const float* proprio;          /* [n][n_proprio] raw robot proprioception, host memory */
  int32_t n_proprio;
  const char* const* prompts;    /* pi05: [n] prompts, or NULL = `prompt` for every episode */
  const char* prompt;
  const int32_t* task;           /* PiBehavior: [n] task id 0..49 */
  const int32_t* stage;          /* PiBehavior: [n] stage input */
  const double* initial_actions; /* PiBehavior: [n][4][23] kept absolute actions, or NULL */
  const uint8_t* use_initial;    /* PiBehavior: [n] 1 = inpaint with initial_actions[i] */
  const float* noise;            /* [n][horizon][32] host start noise, or NULL = the engine stream (episode order) */
} Pi05BatchIn;

typedef struct {
  double* actions;               /* [n][horizon][action_dim] robot units (f64, like pi05_infer), host */
  float* stage_logits;           /* PiBehavior optional [n][15] (invalid stages -inf), host */
} Pi05BatchOut;

typedef struct {
  float total_ms, prefix_ms, suffix_ms, host_ms;
  int32_t n, chunks;             /* episodes, suffix batches (n / max_batch rounded up) */
} Pi05BatchTiming;

typedef struct {
  int32_t n;
  const int32_t* slots;          /* [n] environment slots (wrapper state per slot) */
  const Pi05View* views;         /* [n][3] */
  const float* proprio;          /* [n][n_proprio], host */
  int32_t n_proprio;
  const char* const* prompts;    /* pi05: [n] or NULL = `prompt` */
  const char* prompt;
  int32_t replan_every;          /* pi05 receding horizon (16 for the radio server) */
  const int32_t* task;           /* PiBehavior: [n] task ids (a change resets the slot like pi05_set_task) */
} Pi05ActBatchIn;

/* Like pi05_create plus room for max_batch episodes per suffix batch (KV caches ~19 MB per episode). */
PI05_API Pi05Engine* pi05_create_batch(const char* weights_path, int32_t device, int32_t max_batch, char* err,
                                       int32_t err_len);
/* n inferences (n may exceed max_batch: processed in chunks). Returns 0 or < 0 on error (pi05_last_error). */
PI05_API int32_t pi05_infer_batch(Pi05Engine* e, const Pi05BatchIn* in, const Pi05BatchOut* out,
                                  Pi05BatchTiming* timing);
/* One environment step for n slots: actions_out [n][action_dim] float32 robot units (the wrappers' action dtype),
 * new_chunk [n] (optional) = 1 where a new chunk was inferred this step. */
PI05_API int32_t pi05_act_batch(Pi05Engine* e, const Pi05ActBatchIn* in, float* actions_out, uint8_t* new_chunk,
                                Pi05BatchTiming* timing);
PI05_API int32_t pi05_batch_cap(Pi05Engine* e);

#ifdef __cplusplus
}
#endif
#endif
