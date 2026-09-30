/* DRAFT for review (engine lead / G4): batched inference of the native pi0.5 engine. Not implemented yet.
 *
 * One call = one model inference per episode for n episodes at once (weights read once, rows of all episodes in
 * the same GEMMs). Chunk level only: the B1K wrappers (receding horizon, PiBehavior voting / cubic resampling /
 * correction rules) stay per episode in the caller or in pi05_act.
 *
 * Bit identity: the actions of episode i are bit-identical to pi05_infer / pi05_infer_pb with the same inputs, for
 * every n (the batched kernels keep each row's arithmetic, including split-K partitions, exactly as in the n = 1 path).
 */
#ifndef PI05_BATCH_H
#define PI05_BATCH_H
#include "pi05_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PI05_PIX_RGB8 = 0, PI05_PIX_RGBA8 = 1 } Pi05PixelFormat;

typedef struct {
  const uint8_t* data;    /* first pixel of the view */
  int32_t h, w;           /* 224x224 is used as is; other sizes: resize_with_pad (PIL bilinear, exact) on the GPU */
  int64_t row_stride;     /* bytes between rows */
  int32_t format;         /* Pi05PixelFormat: RGB8, or RGBA8 renderer output (alpha ignored, like obs[..., :3]) */
  int32_t on_device;      /* 1: CUDA device memory on the engine's device (no host copy) */
} Pi05View;

typedef struct {
  int32_t n;                    /* episodes in this call, 1 .. max_batch given at pi05_create_batch */
  const Pi05View* views;        /* [n][3]: head, left wrist, right wrist (Pi05Info.cam_keys order) */
  const float* proprio;         /* [n][n_proprio] raw robot proprioception (evaluator "robot_r1::proprio") */
  int32_t n_proprio;
  int32_t proprio_on_device;    /* 1: proprio is device memory */
  /* openpi pi05 (radio): */
  const char* const* prompts;   /* [n] task prompts, or NULL = the prompt given at create for every episode */
  /* PiBehavior (2025 1st place): */
  const int32_t* task;          /* [n] task id 0..49 */
  const int32_t* stage;         /* [n] stage token input (the wrapper's current stage) */
  const double* initial_actions;/* [n][4][23] kept absolute actions for soft inpainting, or NULL */
  const uint8_t* use_initial;   /* [n] 1 = episode i inpaints with initial_actions[i] (first chunk: 0) */
  /* start noise: */
  const float* noise;           /* [n][horizon][32] f32 device or host (noise_on_device), or NULL = the engine streams */
  int32_t noise_on_device;
  const int32_t* slots;         /* [n] per-episode RNG stream / slot id when noise == NULL (NULL = 0..n-1) */
} Pi05BatchIn;

typedef struct {
  double* actions;              /* [n][horizon][action_dim] robot units (f64, same values as pi05_infer) */
  int32_t actions_on_device;    /* 1: device buffer (the call stays asynchronous on `stream`) */
  float* raw_actions;           /* optional [n][horizon][32] normalized model output (device) */
  float* stage_logits;          /* PiBehavior optional [n][15] (device or host, like actions) */
  int32_t* stage_pred;          /* PiBehavior optional [n] argmax of stage_logits over valid stages */
} Pi05BatchOut;

typedef struct {
  float total_ms, gpu_ms, prep_ms;
  int32_t n, sub_batches;       /* prefix processed in sub-batches of <= prefix_chunk episodes (memory bound) */
} Pi05BatchTiming;

/* Like pi05_create, plus room for up to max_batch episodes per call. prefix_chunk: episodes per prefix sub-batch
 * (0 = choose from free memory). Weights are shared; per-episode memory is the KV cache (~19 MB) plus activations. */
PI05_API Pi05Engine* pi05_create_batch(const char* weights_path, int32_t device, int32_t max_batch,
                                       int32_t prefix_chunk, char* err, int32_t err_len);
/* n inferences. stream: cudaStream_t (NULL = the engine's stream); with device inputs/outputs nothing blocks the
 * host except the small per-episode host preprocessing (state tokens) — returns after enqueueing. */
PI05_API int32_t pi05_infer_batch(Pi05Engine* e, const Pi05BatchIn* in, const Pi05BatchOut* out, void* stream,
                                  Pi05BatchTiming* timing);

#ifdef __cplusplus
}
#endif
#endif
