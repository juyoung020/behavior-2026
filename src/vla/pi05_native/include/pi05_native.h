/* pi0.5 native inference engine -- C API (no Python, no PyTorch, no JAX in the execution path).
 *
 * One engine = one model on the current CUDA device. Not thread-safe; call from one thread.
 *
 * Images: 3 views in openpi order (head, left wrist, right wrist). Each view is H x W pixels of uint8 RGB(A) with
 * explicit row / pixel strides in bytes (RGBA views use only the first three channels, like `obs[..., :3]`).
 * Views that are not 224x224 are resized exactly like openpi_client.image_tools.resize_with_pad (PIL bilinear).
 * `on_device` = 1 means the pointers are CUDA device pointers (e.g. a torch CUDA tensor); they must then be
 * 224x224 already.
 */
#ifndef PI05_NATIVE_H
#define PI05_NATIVE_H
#include <stdint.h>

#if defined(PI05_STATIC)
#define PI05_API
#elif defined(_WIN32)
#ifdef PI05_BUILD
#define PI05_API __declspec(dllexport)
#else
#define PI05_API __declspec(dllimport)
#endif
#else
#define PI05_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Pi05Engine Pi05Engine;

typedef struct {
  const uint8_t* data;
  int32_t h, w;
  int64_t row_stride, pix_stride;
  int32_t on_device;
} Pi05Image;

typedef struct {
  int32_t action_horizon;   /* model chunk length (32 for pi05_b1k) */
  int32_t action_dim;       /* robot action dim returned (23) */
  int32_t proprio_min_len;  /* proprio vector must have at least this many entries */
  int64_t weight_bytes, activation_bytes;
  int32_t model_kind;       /* 0 = openpi pi05 (text prompt), 1 = PiBehavior (2025 1st place: task id + stage) */
  int32_t num_steps;        /* flow-matching denoising steps */
  char robot_name[64];
  char cam_keys[3][96];     /* evaluator observation keys of the 3 cameras */
} Pi05Info;

typedef struct {
  float preprocess_ms, upload_ms, gpu_ms, postprocess_ms, total_ms;
  int32_t tokens;   /* valid prompt tokens */
  int32_t truncated;
} Pi05Timing;

/* Loads weights (.pi05w), captures the CUDA graph. Returns NULL and fills err on failure. */
PI05_API Pi05Engine* pi05_create(const char* weights_path, int32_t device, char* err, int32_t err_len);
PI05_API void pi05_destroy(Pi05Engine* e);
PI05_API void pi05_info(Pi05Engine* e, Pi05Info* out);

/* One openpi Policy.infer: inputs -> action chunk [action_horizon][action_dim] (float64, robot units).
 * noise: [action_horizon][32] float32 start noise, or NULL for the next draw of the openpi server stream
 * (jax.random.key(seed) split once per inference, bit-identical keys; seed 0 = openpi default, see pi05_seed). */
PI05_API int32_t pi05_infer(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                            const char* prompt, const float* noise, double* actions_out, Pi05Timing* timing);

/* openpi B1KPolicyWrapper, control_mode=receding_horizon (eval_b1k_wrapper.py:118-162): per environment slot,
 * infer a chunk, execute `replan_every` actions (16 for the radio server), then infer again.
 * Writes one action (float32 [action_dim]) and returns 1 if a new chunk was inferred this call, 0 if buffered,
 * <0 on error. */
PI05_API int32_t pi05_act(Pi05Engine* e, int32_t slot, const Pi05Image imgs[3], const float* proprio,
                          int32_t n_proprio, const char* prompt, int32_t replan_every, float* action_out,
                          Pi05Timing* timing);
/* ---- PiBehavior (2025 1st place) ----
 * pi05_act() runs the 1st-place B1KPolicyWrapper for these weights: 30 predicted, 26 executed in 20 steps (cubic
 * resampling unless the gripper moves), 4 kept for soft inpainting, stage voting (2 of the last 3 predictions),
 * gripper correction rules. The task id must be set per slot (evaluator obs "task_id"). */
PI05_API void pi05_set_task(Pi05Engine* e, int32_t slot, int32_t task);
/* Stage input of the model. mode 0: the model's own voting (default; stage >= 0 also resets the current stage),
 * mode 1: the stage is fixed from outside (a planner drives the policy), voting is off until mode 0. */
PI05_API void pi05_set_stage(Pi05Engine* e, int32_t slot, int32_t stage, int32_t mode);
PI05_API void pi05_get_stage(Pi05Engine* e, int32_t slot, int32_t* stage, int32_t* predicted, int32_t* forced);
PI05_API void pi05_pb_config(Pi05Engine* e, int32_t actions_to_execute, int32_t actions_to_keep,
                             int32_t execute_in_n_steps, int32_t apply_eval_tricks);
/* One model call. initial_actions: [n_initial][23] absolute robot actions kept from the previous chunk (or NULL);
 * noise: [30][32] start noise used as given when there is no inpainting (NULL = correlated noise from the stream).
 * actions_out [30][23] float64 robot units, stage_logits_out [15] (invalid stages = -inf). */
PI05_API int32_t pi05_infer_pb(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                               int32_t task, int32_t stage, const double* initial_actions, int32_t n_initial,
                               const float* noise, double* actions_out, float* stage_logits_out, Pi05Timing* timing);
PI05_API void pi05_reset(Pi05Engine* e, int32_t slot); /* slot < 0: every slot */
PI05_API void pi05_seed(Pi05Engine* e, uint64_t seed);
PI05_API const char* pi05_last_error(Pi05Engine* e);

#ifdef __cplusplus
}
#endif
#endif
