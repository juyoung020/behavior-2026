// Trainable parameter set (f32 masters, gradients, AdamW moments, EMA; optional host offload of the optimizer state)
// and frozen bf16 weights, both read from a training state container (p.<openpi path> f32, f.<openpi path> bf16).
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../../pi05_native/src/weights.h"
#include "tkern.cuh"

namespace pi05t {

struct TParam {
  std::string name;              // openpi path, '/'-joined (e.g. PaliGemma/llm/layers/attn/q_einsum_1/w)
  std::vector<int64_t> shape;
  long long n = 0;
  float *p = nullptr, *g = nullptr, *m = nullptr, *v = nullptr, *ema = nullptr;  // f32 master, grad, Adam, EMA
  bool host_mv = false, host_ema = false;  // m, v / ema live in pinned host memory (streamed through the GPU per step)
  bf16* pb = nullptr;            // bf16 working copy (params used through .astype(bf16))
  bool round_grad = false;       // JAX gradient is a bf16 result (bf16 einsum / bf16 Dense), cast to f32 once
};

struct OptCfg {
  int warmup = 1000, decay_steps = 30000;
  double peak = 2.5e-5, end = 2.5e-6;
  double b1 = 0.9, b2 = 0.95, eps = 1e-8, wd = 1e-10, clip = 1.0, ema = 0.99;
  bool use_ema = true;
  double lr(int count) const;  // optax.warmup_cosine_decay_schedule(peak/(warmup+1), peak, warmup, decay_steps, end)
  void read(const pi05::WeightFile& wf);
};

class DevMem {  // owns device / pinned host allocations
 public:
  ~DevMem();
  template <class T>
  T* dev(size_t n);
  float* host(long long n);
 private:
  std::vector<void*> d_, h_;
};

class ParamSet {
 public:
  // needs_bf16(name) / round_grad(name) decide the working copy and the gradient rounding per parameter
  bool load(const pi05::WeightFile& wf, int offload, const OptCfg& opt, DevMem& mem,
            const std::function<bool(const std::string&)>& needs_bf16,
            const std::function<bool(const std::string&)>& round_grad, cudaStream_t st, std::string* err);
  TParam* get(const std::string& name);
  void zero_grads(cudaStream_t st);
  void finalize(cudaStream_t st);
  double grad_norm(cudaStream_t st);
  void step(const OptCfg& opt, int count, cudaStream_t st);  // clip -> AdamW -> EMA -> refresh bf16 copies
  std::vector<TParam>& all() { return ps_; }
  long long numel() const;
 private:
  std::vector<TParam> ps_;
  std::map<std::string, size_t> idx_;
  double* d_sumsq_ = nullptr;
  float *sm_ = nullptr, *sv_ = nullptr, *se_ = nullptr;
  static constexpr long long CHUNK = 16LL << 20;
};

class FrozenSet {  // f.<name> bf16 tensors on the GPU
 public:
  bool load(const pi05::WeightFile& wf, DevMem& mem, std::string* err);
  const bf16* get(const std::string& name) const;
  size_t bytes = 0;
 private:
  std::map<std::string, const bf16*> w_;
};

}  // namespace pi05t
