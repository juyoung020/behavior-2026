#include "tparams.h"

#include <cmath>
#include <cstring>

namespace pi05t {

double OptCfg::lr(int c) const {  // optax, f32 arithmetic
  const float pk = (float)peak;
  if (c < warmup) {
    const float init = (float)(peak / (warmup + 1));
    const float frac = 1.0f - (float)c / (float)warmup;
    return (init - pk) * frac + pk;
  }
  const int ds = decay_steps - warmup;
  const int cc = std::min(c - warmup, ds);
  const float cosd = 0.5f * (1.0f + cosf(3.14159265358979323846f * (float)cc / (float)ds));
  const float alpha = (float)(end / peak);
  return pk * ((1.0f - alpha) * cosd + alpha);
}

void OptCfg::read(const pi05::WeightFile& wf) {
  auto d = [&](const char* k, double def) {
    const auto* v = wf.cfg(k);
    return v && !v->empty() ? std::stod((*v)[0]) : def;
  };
  warmup = (int)d("lr.warmup", 1000);
  peak = d("lr.peak", 2.5e-5);
  decay_steps = (int)d("lr.decay_steps", 30000);
  end = d("lr.end", 2.5e-6);
  b1 = d("adam.b1", 0.9); b2 = d("adam.b2", 0.95); eps = d("adam.eps", 1e-8);
  wd = d("adam.wd", 1e-10); clip = d("clip", 1.0);
  const auto* ev = wf.cfg("ema");
  use_ema = ev && !ev->empty() && (*ev)[0] != "None";
  ema = use_ema ? std::stod((*ev)[0]) : 0.0;
}

DevMem::~DevMem() {
  for (void* p : d_) cudaFree(p);
  for (void* p : h_) cudaFreeHost(p);
}
template <class T>
T* DevMem::dev(size_t n) {
  void* p = nullptr;
  PI05_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  PI05_CUDA(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  // the memset runs on the legacy default stream, which does not order with our non-blocking streams: finish it
  // before anyone writes the buffer from another stream
  PI05_CUDA(cudaDeviceSynchronize());
  d_.push_back(p);
  return (T*)p;
}
template float* DevMem::dev<float>(size_t);
template bf16* DevMem::dev<bf16>(size_t);
template double* DevMem::dev<double>(size_t);
template float2* DevMem::dev<float2>(size_t);
template int* DevMem::dev<int>(size_t);
template uint8_t* DevMem::dev<uint8_t>(size_t);
float* DevMem::host(long long n) {
  void* p = nullptr;
  PI05_CUDA(cudaMallocHost(&p, std::max<long long>(n, 1) * 4));
  memset(p, 0, std::max<long long>(n, 1) * 4);
  h_.push_back(p);
  return (float*)p;
}

bool ParamSet::load(const pi05::WeightFile& wf, int offload, const OptCfg& opt, DevMem& mem,
                    const std::function<bool(const std::string&)>& needs_bf16,
                    const std::function<bool(const std::string&)>& round_grad, cudaStream_t st, std::string* err) {
  std::vector<float> host;
  for (const pi05::TensorInfo& t : wf.tensors()) {
    if (t.name.rfind("p.", 0) != 0) continue;
    if (t.dtype != pi05::DType::F32) { *err = "trainable param not f32: " + t.name; return false; }
    TParam P;
    P.name = t.name.substr(2);
    P.shape = t.shape;
    P.n = t.numel();
    host.resize(P.n);
    if (!wf.read(t, host.data(), err)) return false;
    P.p = mem.dev<float>(P.n);
    P.g = mem.dev<float>(P.n);
    P.host_mv = offload >= 2;
    P.host_ema = offload >= 1;
    P.m = P.host_mv ? mem.host(P.n) : mem.dev<float>(P.n);
    P.v = P.host_mv ? mem.host(P.n) : mem.dev<float>(P.n);
    if (opt.use_ema) P.ema = P.host_ema ? mem.host(P.n) : mem.dev<float>(P.n);
    PI05_CUDA(cudaMemcpy(P.p, host.data(), P.n * 4, cudaMemcpyHostToDevice));
    if (P.ema) PI05_CUDA(cudaMemcpy(P.ema, host.data(), P.n * 4, cudaMemcpyDefault));
    P.round_grad = round_grad(P.name);
    if (needs_bf16(P.name)) {
      P.pb = mem.dev<bf16>(P.n);
      f32_to_bf16(P.p, P.pb, P.n, st);
    }
    idx_[P.name] = ps_.size();
    ps_.push_back(std::move(P));
  }
  d_sumsq_ = mem.dev<double>(1);
  if (offload >= 1) se_ = mem.dev<float>(CHUNK);
  if (offload >= 2) { sm_ = mem.dev<float>(CHUNK); sv_ = mem.dev<float>(CHUNK); }
  return true;
}

TParam* ParamSet::get(const std::string& name) {
  auto it = idx_.find(name);
  return it == idx_.end() ? nullptr : &ps_[it->second];
}
long long ParamSet::numel() const {
  long long n = 0;
  for (auto& P : ps_) n += P.n;
  return n;
}
void ParamSet::zero_grads(cudaStream_t st) {
  for (auto& P : ps_) PI05_CUDA(cudaMemsetAsync(P.g, 0, P.n * 4, st));
}
void ParamSet::finalize(cudaStream_t st) {
  for (auto& P : ps_)
    if (P.round_grad) round_bf16_inplace(P.g, P.n, st);
}
double ParamSet::grad_norm(cudaStream_t st) {
  PI05_CUDA(cudaMemsetAsync(d_sumsq_, 0, 8, st));
  for (auto& P : ps_) sumsq_f32(P.g, P.n, d_sumsq_, st);
  double s = 0;
  PI05_CUDA(cudaMemcpyAsync(&s, d_sumsq_, 8, cudaMemcpyDeviceToHost, st));
  PI05_CUDA(cudaStreamSynchronize(st));
  return std::sqrt(s);
}
void ParamSet::step(const OptCfg& opt, int count, cudaStream_t st) {
  const float gn = (float)grad_norm(st);
  const float lr = (float)opt.lr(count);
  const int t = count + 1;
  const float bc1 = 1.0f - powf((float)opt.b1, (float)t), bc2 = 1.0f - powf((float)opt.b2, (float)t);
  const auto H2D = cudaMemcpyHostToDevice, D2H = cudaMemcpyDeviceToHost;
  for (auto& P : ps_) {
    for (long long o = 0; o < P.n; o += CHUNK) {  // host-resident state streamed chunk by chunk (same arithmetic)
      const long long n = std::min(CHUNK, P.n - o);
      float *m = P.m + o, *v = P.v + o, *e = P.ema ? P.ema + o : nullptr;
      if (P.host_mv) {
        PI05_CUDA(cudaMemcpyAsync(sm_, m, n * 4, H2D, st));
        PI05_CUDA(cudaMemcpyAsync(sv_, v, n * 4, H2D, st));
        m = sm_;
        v = sv_;
      }
      if (e && P.host_ema) {
        PI05_CUDA(cudaMemcpyAsync(se_, e, n * 4, H2D, st));
        e = se_;
      }
      adamw_step(P.p + o, P.g + o, m, v, n, gn, (float)opt.clip, (float)opt.b1, (float)(1.0 - opt.b1), (float)opt.b2,
                 (float)(1.0 - opt.b2), (float)opt.eps, (float)opt.wd, lr, bc1, bc2, st);
      if (e) ema_step(e, P.p + o, n, (float)opt.ema, (float)(1.0 - opt.ema), st);
      if (P.host_mv) {
        PI05_CUDA(cudaMemcpyAsync(P.m + o, sm_, n * 4, D2H, st));
        PI05_CUDA(cudaMemcpyAsync(P.v + o, sv_, n * 4, D2H, st));
      }
      if (e && P.host_ema) PI05_CUDA(cudaMemcpyAsync(P.ema + o, se_, n * 4, D2H, st));
    }
    if (P.pb) f32_to_bf16(P.p, P.pb, P.n, st);
  }
  PI05_CUDA(cudaStreamSynchronize(st));
}

bool FrozenSet::load(const pi05::WeightFile& wf, DevMem& mem, std::string* err) {
  for (const pi05::TensorInfo& t : wf.tensors()) {
    if (t.name.rfind("f.", 0) != 0) continue;
    if (t.dtype != pi05::DType::BF16) { *err = "frozen param not bf16: " + t.name; return false; }
    std::vector<uint8_t> h(t.nbytes);
    if (!wf.read(t, h.data(), err)) return false;
    bf16* d = mem.dev<bf16>(t.numel());
    PI05_CUDA(cudaMemcpy(d, h.data(), t.nbytes, cudaMemcpyHostToDevice));
    w_[t.name.substr(2)] = d;
    bytes += t.nbytes;
  }
  return true;
}
const bf16* FrozenSet::get(const std::string& name) const {
  auto it = w_.find(name);
  return it == w_.end() ? nullptr : it->second;
}

}  // namespace pi05t
