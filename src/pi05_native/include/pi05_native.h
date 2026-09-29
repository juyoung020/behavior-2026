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
 * noise: [action_horizon][32] float32 start noise, or NULL to draw from the engine's generator (seeded). */
PI05_API int32_t pi05_infer(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                            const char* prompt, const float* noise, double* actions_out, Pi05Timing* timing);

/* openpi B1KPolicyWrapper, control_mode=receding_horizon (eval_b1k_wrapper.py:118-162): per environment slot,
 * infer a chunk, execute `replan_every` actions (16 for the radio server), then infer again.
 * Writes one action (float32 [action_dim]) and returns 1 if a new chunk was inferred this call, 0 if buffered,
 * <0 on error. */
PI05_API int32_t pi05_act(Pi05Engine* e, int32_t slot, const Pi05Image imgs[3], const float* proprio,
                          int32_t n_proprio, const char* prompt, int32_t replan_every, float* action_out,
                          Pi05Timing* timing);
PI05_API void pi05_reset(Pi05Engine* e, int32_t slot); /* slot < 0: every slot */
PI05_API void pi05_seed(Pi05Engine* e, uint64_t seed);
PI05_API const char* pi05_last_error(Pi05Engine* e);

#ifdef __cplusplus
}
#endif
#endif
