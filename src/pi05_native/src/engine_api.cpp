// C API: input transforms -> CUDA graph -> output transforms, plus the B1K receding-horizon wrapper.
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
#include "model.h"
#include "tokenizer.h"
#include "weights.h"

using namespace pi05;

namespace {
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Slot {  // B1KPolicyWrapper state for one environment
  bool has = false;
  int step = 0, idx = 0, len = 0;
  std::vector<float> buf;  // [len][action_dim] float32 (the wrapper's action_buffer dtype)
};

// splitmix64 + Box-Muller: standard normal start noise when the caller gives none
struct Rng {
  uint64_t s = 0;
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double uni() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
  void normal(float* out, int n) {
    for (int i = 0; i < n; i += 2) {
      double r = std::sqrt(-2.0 * std::log(uni())), t = 6.283185307179586 * uni();
      out[i] = (float)(r * std::cos(t));
      if (i + 1 < n) out[i + 1] = (float)(r * std::sin(t));
    }
  }
};
}  // namespace

struct Pi05Engine {
  WeightFile wf;
  Tokenizer tok;
  RobotSpec rs;
  Model model;
  cudaStream_t st = nullptr;
  uint8_t* img_host = nullptr;  // pinned [3][224][224][3]
  std::vector<float> raw, noise;
  Rng rng;
  std::map<int, Slot> slots;
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
  const TensorInfo* t = e->wf.find("tokenizer.model");
  if (!t) return fail("no tokenizer in weight file");
  std::vector<uint8_t> tm(t->nbytes);
  if (!e->wf.read(*t, tm.data(), &m) || !e->tok.load(tm.data(), tm.size(), &m)) return fail("tokenizer: " + m);
  if (!load_robot_spec(e->wf, &e->rs, &m)) return fail(m);
  if (!e->model.load(e->wf, &m)) return fail("model: " + m);
  PI05_CUDA(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking));
  PI05_CUDA(cudaMallocHost(&e->img_host, 3 * 224 * 224 * 3));
  e->raw.resize((size_t)e->model.cfg.ah * e->model.cfg.ad);
  e->noise.resize(e->raw.size());
  // capture the whole forward once (shapes are fixed; prompt length lives in device memory)
  std::vector<int> dummy_tokens(16, 2);
  memset(e->img_host, 0, 3 * 224 * 224 * 3);
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
  out->proprio_min_len = e->rs.max_proprio_index() + 1;
  out->weight_bytes = (int64_t)e->model.weight_bytes;
  out->activation_bytes = (int64_t)e->model.act_bytes;
  snprintf(out->robot_name, sizeof out->robot_name, "%s", e->rs.name.c_str());
  for (int i = 0; i < 3; ++i) snprintf(out->cam_keys[i], sizeof out->cam_keys[i], "%s", e->rs.cams[i].c_str());
}

PI05_API int32_t pi05_infer(Pi05Engine* e, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                            const char* prompt, const float* noise, double* actions_out, Pi05Timing* timing) {
  auto t0 = clk::now();
  if (n_proprio < e->rs.max_proprio_index() + 1) { e->err = "proprio too short"; return -1; }
  bool dev = false;
  if (!stage_images(e, imgs, &dev)) return -1;
  PreparedInput in;
  prepare_input(e->rs, e->tok, proprio, n_proprio, prompt ? prompt : "", e->model.cfg.max_tok, e->model.cfg.ad, &in);
  if (noise) memcpy(e->noise.data(), noise, e->noise.size() * 4);
  else e->rng.normal(e->noise.data(), (int)e->noise.size());
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

PI05_API int32_t pi05_act(Pi05Engine* e, int32_t slot, const Pi05Image imgs[3], const float* proprio, int32_t n_proprio,
                          const char* prompt, int32_t replan_every, float* action_out, Pi05Timing* timing) {
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

PI05_API void pi05_reset(Pi05Engine* e, int32_t slot) {
  if (slot < 0) e->slots.clear();
  else e->slots.erase(slot);
}

PI05_API void pi05_seed(Pi05Engine* e, uint64_t seed) { e->rng.s = seed; }

PI05_API const char* pi05_last_error(Pi05Engine* e) { return e->err.c_str(); }

}  // extern "C"
