// C API: input transforms -> CUDA graph -> output transforms, plus the B1K wrappers
// (openpi receding horizon for pi05; the 1st-place PiBehavior wrapper with stage voting / compression / inpainting).
#define PI05_BUILD
#include "../include/pi05_native.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "host_io.h"
#include "image.h"
#include "jax_rng.h"
#include "model.h"
#include "pb_host.h"
#include "tokenizer.h"
#include "weights.h"

using namespace pi05;

namespace {
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Slot {  // openpi B1KPolicyWrapper state for one environment
  bool has = false;
  int step = 0, idx = 0, len = 0;
  std::vector<float> buf;  // [len][action_dim] float32 (the wrapper's action_buffer dtype)
};

}  // namespace

struct Pi05Engine {
  WeightFile wf;
  Tokenizer tok;
  RobotSpec rs;
  PbSpec pb;
  PbWrapperCfg pbw;
  bool is_pb = false;
  Model model;
  cudaStream_t st = nullptr;
  uint8_t* img_host = nullptr;  // pinned [3][224][224][3]
  std::vector<float> raw, noise;
  JaxKey rng = jax_key(0);  // openpi Policy: self._rng = jax.random.key(0)
  std::map<int, Slot> slots;
  std::map<int, PbSlot> pb_slots;
  std::string err;
};

static bool stage_images(Pi05Engine* e, const Pi05Image imgs[3], bool* device_done) {
  const size_t view = 224 * 224 * 3;
  *device_done = false;
  for (int i = 0; i < 3; ++i) {
    const Pi05Image& im = imgs[i];
    if (!im.data) { e->err = "missing image"; return false; }
    if (im.on_device) {
      if (im.h != 224 || im.w != 224 || im.row_stride != 224 * im.pix_stride) {
        e->err = "device images must be contiguous 224x224";
        return false;
      }
      // strided RGB(A) -> packed RGB on the device
      PI05_CUDA(cudaMemcpy2DAsync(e->model.d_img + i * view, 3, im.data, (size_t)im.pix_stride, 3, 224 * 224,
                                  cudaMemcpyDeviceToDevice, e->st));
      *device_done = true;
      continue;
    }
    ImageView v{im.data, im.h, im.w, im.row_stride, im.pix_stride};
    if (im.h == 224 && im.w == 224 && im.pix_stride == 3 && im.row_stride == 224 * 3) {
      memcpy(e->img_host + i * view, im.data, view);
    } else {
      resize_with_pad(v, 224, 224, e->img_host + i * view);
    }
  }
  return true;
}

extern "C" {

PI05_API Pi05Engine* pi05_create(const char* weights_path, int32_t device, char* err, int32_t err_len) {
  auto fail = [&](const std::string& m) -> Pi05Engine* {
    if (err && err_len > 0) snprintf(err, err_len, "%s", m.c_str());
    return nullptr;
  };
  if (cudaSetDevice(device) != cudaSuccess) return fail("cudaSetDevice failed");
  auto e = std::make_unique<Pi05Engine>();
  std::string m;
  if (!e->wf.open(weights_path, &m)) return fail(m);
  e->is_pb = e->wf.cfg_str("model", "pi05") == "pi_behavior";
  if (e->is_pb) {
    if (!load_pb_spec(e->wf, &e->pb, &m)) return fail(m);
    e->rs.name = e->wf.cfg_str("robot.name", "robot_r1");
    e->rs.action_dim = 23;
    for (int i = 0; i < 3; ++i) e->rs.cams.push_back(e->wf.cfg_str("robot.cam" + std::to_string(i), ""));
  } else {
    const TensorInfo* t = e->wf.find("tokenizer.model");
    if (!t) return fail("no tokenizer in weight file");
    std::vector<uint8_t> tm(t->nbytes);
    if (!e->wf.read(*t, tm.data(), &m) || !e->tok.load(tm.data(), tm.size(), &m)) return fail("tokenizer: " + m);
    if (!load_robot_spec(e->wf, &e->rs, &m)) return fail(m);
  }
  if (!e->model.load(e->wf, &m)) return fail("model: " + m);
  PI05_CUDA(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking));
  PI05_CUDA(cudaMallocHost(&e->img_host, 3 * 224 * 224 * 3));
  e->raw.resize((size_t)e->model.cfg.ah * e->model.cfg.ad);
  e->noise.resize(e->raw.size());
  // capture the whole forward once (shapes are fixed; prompt length, task and stage live in device memory)
  std::vector<int> dummy_tokens(e->is_pb ? 32 : 16, e->is_pb ? 128 : 2);
  memset(e->img_host, 0, 3 * 224 * 224 * 3);
  e->model.set_pb_inputs(0, 0, nullptr, nullptr, 0, e->st);
  e->model.upload_inputs(e->img_host, dummy_tokens, e->noise.data(), e->st);
  if (!e->model.capture_graph(e->st, &m)) return fail("graph: " + m);
  e->model.forward(e->st);  // warm up
  PI05_CUDA(cudaStreamSynchronize(e->st));
  return e.release();
}

PI05_API void pi05_destroy(Pi05Engine* e) {
  if (!e) return;
  if (e->img_host) cudaFreeHost(e->img_host);
  if (e->st) cudaStreamDestroy(e->st);
  delete e;
}

PI05_API void pi05_info(Pi05Engine* e, Pi05Info* out) {
  memset(out, 0, sizeof *out);
  out->action_horizon = e->model.cfg.ah;
  out->action_dim = e->rs.action_dim;
  out->proprio_min_len = e->is_pb ? 57 : e->rs.max_proprio_index() + 1;
  out->weight_bytes = (int64_t)e->model.weight_bytes;
  out->activation_bytes = (int64_t)e->model.act_bytes;
  out->model_kind = e->is_pb ? 1 : 0;
  out->num_steps = e->model.cfg.steps;
  snprintf(out->robot_name, sizeof out->robot_name, "%s", e->rs.name.c_str());
  for (int i = 0; i < 3; ++i) snprintf(out->cam_keys[i], sizeof out->cam_keys[i], "%s", e->rs.cams[i].c_str());
}

PI05_API int32_t pi05_infer(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                            const char* prompt, const float* noise, double* actions_out, Pi05Timing* timing) {
  if (e->is_pb) { e->err = "PiBehavior weights: use pi05_infer_pb"; return -1; }
  auto t0 = clk::now();
  if (n_proprio < e->rs.max_proprio_index() + 1) { e->err = "proprio too short"; return -1; }
  bool dev = false;
  if (!stage_images(e, imgs, &dev)) return -1;
  PreparedInput in;
  prepare_input(e->rs, e->tok, proprio, n_proprio, prompt ? prompt : "", e->model.cfg.max_tok, e->model.cfg.ad, &in);
  if (noise) memcpy(e->noise.data(), noise, e->noise.size() * 4);
  else {  // same stream as the openpi server: rng, sample = split(rng); normal(sample, (1, ah, ad))
    JaxKey sample;
    jax_split(e->rng, &e->rng, &sample);
    jax_normal(sample, e->noise.data(), (int)e->noise.size());
  }
  const double t_pre = ms_since(t0);
  auto t1 = clk::now();
  // device images were already copied into d_img on the stream; host images go through upload_inputs
  e->model.upload_inputs(dev ? nullptr : e->img_host, in.tokens, e->noise.data(), e->st);
  const double t_up = ms_since(t1);
  auto t2 = clk::now();
  e->model.forward(e->st);
  e->model.download_actions(e->raw.data(), e->st);
  const double t_gpu = ms_since(t2);
  auto t3 = clk::now();
  postprocess_actions(e->rs, e->raw.data(), e->model.cfg.ah, e->model.cfg.ad, in.state_f32, actions_out);
  if (timing) {
    timing->preprocess_ms = (float)t_pre;
    timing->upload_ms = (float)t_up;
    timing->gpu_ms = (float)t_gpu;
    timing->postprocess_ms = (float)ms_since(t3);
    timing->total_ms = (float)ms_since(t0);
    timing->tokens = (int32_t)in.tokens.size();
    timing->truncated = in.truncated;
  }
  return 0;
}

// PiBehaviorPolicy.infer (pi_behavior_policy.py) + sample_actions (pi_behavior.py:905-1118)
PI05_API int32_t pi05_infer_pb(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                               int32_t task, int32_t stage, const double* initial_actions, int32_t n_initial,
                               const float* noise, double* actions_out, float* stage_logits_out, Pi05Timing* timing) {
  if (!e->is_pb) { e->err = "pi05_infer_pb needs PiBehavior weights"; return -1; }
  auto t0 = clk::now();
  const PbSpec& s = e->pb;
  if (task < 0 || task >= (int)s.stages.size()) { e->err = "task id out of range"; return -1; }
  if (n_proprio != 23 && n_proprio < 57) { e->err = "proprio too short"; return -1; }
  bool dev = false;
  if (!stage_images(e, imgs, &dev)) return -1;
  float st23[23], norm32[32];
  std::vector<int> tokens(32);
  pb_extract_state(proprio, n_proprio, st23);
  pb_state_tokens(s, st23, norm32, tokens.data());
  const int n = s.ah * s.ad;
  if (noise && !initial_actions) {
    memcpy(e->noise.data(), noise, (size_t)n * 4);  // explicit noise is used as given (no correlation)
  } else {
    // policy: rng, sample_rng = split(rng); sample_actions: rng, noise_rng = split(sample_rng);
    // noise = normal(noise_rng, (1, ah*ad)) @ L^T
    JaxKey sample, r2, nk;
    jax_split(e->rng, &e->rng, &sample);
    jax_split(sample, &r2, &nk);
    std::vector<float> z(n);
    jax_normal(nk, z.data(), n);
    pb_correlate(s, z.data(), e->noise.data());
  }
  std::vector<float> x0O;
  const int keep = n_initial;
  if (initial_actions && keep > 0) {
    x0O.resize((size_t)keep * s.ad);
    pb_initial_actions(s, initial_actions, keep, st23, x0O.data());
  }
  const double t_pre = ms_since(t0);
  auto t1 = clk::now();
  e->model.set_pb_inputs(task, stage, x0O.empty() ? nullptr : x0O.data(), x0O.empty() ? nullptr : e->noise.data(),
                         keep * s.ad, e->st);
  e->model.upload_inputs(dev ? nullptr : e->img_host, tokens, e->noise.data(), e->st);
  const double t_up = ms_since(t1);
  auto t2 = clk::now();
  e->model.forward(e->st);
  e->model.download_actions(e->raw.data(), e->st);
  float logits[15];
  e->model.download_stage_logits(logits, e->st);
  const double t_gpu = ms_since(t2);
  auto t3 = clk::now();
  pb_postprocess(s, e->raw.data(), norm32, actions_out);
  for (int i = 0; i < 15; ++i) {  // invalid stages of this task masked to -inf (pi_behavior.py:1016-1022)
    if (i >= s.stages[task]) logits[i] = -INFINITY;
    if (stage_logits_out) stage_logits_out[i] = logits[i];
  }
  if (timing) {
    timing->preprocess_ms = (float)t_pre;
    timing->upload_ms = (float)t_up;
    timing->gpu_ms = (float)t_gpu;
    timing->postprocess_ms = (float)ms_since(t3);
    timing->total_ms = (float)ms_since(t0);
    timing->tokens = 32;
    timing->truncated = 0;
  }
  return 0;
}

// eval_b1k_wrapper.py B1KPolicyWrapper.act (alstar8 2026 fork), one environment slot
static int32_t pb_act(Pi05Engine* e, int32_t slot, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                      float* action_out, Pi05Timing* timing) {
  PbSlot& s = e->pb_slots[slot];
  const PbWrapperCfg& c = e->pbw;
  if (s.task < 0) { e->err = "task not set (pi05_set_task)"; return -1; }
  float st23[23];
  pb_extract_state(proprio, n_proprio, st23);
  bool inferred = false;
  if (s.n_last == 0 || s.index >= c.execute_in_n_steps) {
    const int ah = e->pb.ah;
    std::vector<double> a((size_t)ah * 23);
    float logits[15];
    const int stage_in = s.forced_stage >= 0 ? s.forced_stage : s.stage;
    const int keep = s.next_init.empty() ? 0 : (int)s.next_init.size() / 23;
    if (pi05_infer_pb(e, imgs, proprio, n_proprio, s.task, stage_in, keep ? s.next_init.data() : nullptr, keep, nullptr,
                      a.data(), logits, timing) != 0)
      return -1;
    bool compress = c.execute_in_n_steps < c.actions_to_execute;
    if (c.apply_eval_tricks) {
      const int corrected = pb_correction_rules(s.task, s.stage, st23, a, ah);
      if (corrected != s.stage) {
        s.stage = corrected;
        s.history.clear();
      }
      if (compress && pb_gripper_variation(a, c.actions_to_execute)) compress = false;
    }
    const int ate = compress ? c.actions_to_execute : c.execute_in_n_steps;
    if (c.actions_to_keep <= 0 || ah < ate + c.actions_to_keep) s.next_init.clear();
    else s.next_init.assign(a.begin() + (size_t)ate * 23, a.begin() + (size_t)(ate + c.actions_to_keep) * 23);
    std::vector<double> last(a.begin(), a.begin() + (size_t)ate * 23);
    if (compress) {
      last = pb_cubic_resample(last, ate, c.execute_in_n_steps);
      const double factor = (double)ate / c.execute_in_n_steps;
      for (int t = 0; t < c.execute_in_n_steps; ++t)
        for (int d = 0; d < 3; ++d) last[(size_t)t * 23 + d] *= factor;  // base velocities scaled
    }
    s.last = last;
    s.n_last = (int)last.size() / 23;
    s.index = 0;
    s.predictions += 1;
    int pred = 0;
    for (int i = 1; i < 15; ++i)
      if (logits[i] > logits[pred]) pred = i;
    s.last_pred_stage = pred;
    if (s.forced_stage < 0) pb_vote(s, e->pb.stages[s.task] - 1, pred, c);
    inferred = true;
  } else if (timing) {
    memset(timing, 0, sizeof *timing);
  }
  if (s.index >= s.n_last) s.index = 0;
  for (int d = 0; d < 23; ++d) action_out[d] = (float)s.last[(size_t)s.index * 23 + d];
  s.index += 1;
  s.step += 1;
  return inferred ? 1 : 0;
}

PI05_API int32_t pi05_act(Pi05Engine* e, int32_t slot, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                          const char* prompt, int32_t replan_every, float* action_out, Pi05Timing* timing) {
  if (e->is_pb) return pb_act(e, slot, imgs, proprio, n_proprio, action_out, timing);
  Slot& s = e->slots[slot];
  const int ad = e->rs.action_dim, ah = e->model.cfg.ah;
  // eval_b1k_wrapper.py:126-131: first call, chunk exhausted, or every `action_horizon` steps
  const bool need = !s.has || s.idx >= s.len || (s.step % replan_every) == 0;
  if (need) {
    std::vector<double> a((size_t)ah * ad);
    if (pi05_infer(e, imgs, proprio, n_proprio, prompt, nullptr, a.data(), timing) != 0) return -1;
    s.len = ah;  // min(target_action.shape[1], max_len) with max_len = model action_horizon
    s.buf.resize((size_t)ah * ad);
    for (size_t i = 0; i < a.size(); ++i) s.buf[i] = (float)a[i];
    s.idx = 0;
    s.has = true;
  } else if (timing) {
    memset(timing, 0, sizeof *timing);
  }
  memcpy(action_out, &s.buf[(size_t)s.idx * ad], ad * 4);
  s.idx += 1;
  s.step += 1;
  return need ? 1 : 0;
}

PI05_API void pi05_set_task(Pi05Engine* e, int32_t slot, int32_t task) {
  PbSlot& s = e->pb_slots[slot];
  if (s.task == task) return;
  // _handle_task_change: stage 0, history, actions and inpainting cleared (eval_b1k_wrapper.py:94-114)
  const int forced = s.forced_stage;
  s = PbSlot{};
  s.task = task;
  s.forced_stage = forced;
}

PI05_API void pi05_set_stage(Pi05Engine* e, int32_t slot, int32_t stage, int32_t mode) {
  PbSlot& s = e->pb_slots[slot];
  if (mode == 1) {
    s.forced_stage = stage;
    s.stage = stage;
    s.history.clear();
  } else {
    s.forced_stage = -1;
    if (stage >= 0) {
      s.stage = stage;
      s.history.clear();
    }
  }
}

PI05_API void pi05_get_stage(Pi05Engine* e, int32_t slot, int32_t* stage, int32_t* predicted, int32_t* forced) {
  PbSlot& s = e->pb_slots[slot];
  if (stage) *stage = s.forced_stage >= 0 ? s.forced_stage : s.stage;
  if (predicted) *predicted = s.last_pred_stage;
  if (forced) *forced = s.forced_stage >= 0;
}

PI05_API void pi05_pb_config(Pi05Engine* e, int32_t actions_to_execute, int32_t actions_to_keep,
                             int32_t execute_in_n_steps, int32_t apply_eval_tricks) {
  e->pbw.actions_to_execute = actions_to_execute;
  e->pbw.actions_to_keep = actions_to_keep;
  e->pbw.execute_in_n_steps = execute_in_n_steps;
  e->pbw.apply_eval_tricks = apply_eval_tricks != 0;
}

PI05_API void pi05_reset(Pi05Engine* e, int32_t slot) {
  if (slot < 0) {
    e->slots.clear();
    for (auto& kv : e->pb_slots) {  // keep the task and any forced stage, clear the episode state
      PbSlot n;
      n.task = kv.second.task;
      n.forced_stage = kv.second.forced_stage;
      if (n.forced_stage >= 0) n.stage = n.forced_stage;
      kv.second = n;
    }
  } else {
    e->slots.erase(slot);
    auto it = e->pb_slots.find(slot);
    if (it != e->pb_slots.end()) {
      PbSlot n;
      n.task = it->second.task;
      n.forced_stage = it->second.forced_stage;
      if (n.forced_stage >= 0) n.stage = n.forced_stage;
      it->second = n;
    }
  }
}

PI05_API void pi05_seed(Pi05Engine* e, uint64_t seed) { e->rng = jax_key(seed); }

PI05_API const char* pi05_last_error(Pi05Engine* e) { return e->err.c_str(); }

}  // extern "C"
