// pi05_train: the native training loop (openpi scripts/train.py without JAX / PyTorch / Python).
//   data     src/vla/fasttrain native loader through its C API (GPU batches, same shuffle order as openpi's loader)
//   random   the openpi key chain (trng.h): fold_in(train_rng, step) -> augmentation keys, noise, beta flow time
//   augment  augmax crop/resize/rotate + color jitter (augment.cu), preprocess_observation(train=True)
//   step     per-sample forward/backward, clip -> AdamW -> EMA (trainer.cu / lora.cu)
//   save     params (+ EMA, Adam state, step) every --save-every steps as a training state container
//
//   pi05_train --state S.pi05d --model M.pi05w --table DIR --lut LUT [--steps N] [--batch B] [--offload 0|1|2]
//              [--seed 42] [--out DIR] [--save-every 1000] [--log-every 10] [--no-aug] [--threads 4]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../../fasttrain/csrc/dlpack_min.h"
#include "../src/augment.cuh"
#include "../src/lora.h"
#include "../src/trainer.h"

extern "C" {
const char* ft_last_error(void);
void* ft_loader_create(const char* table_dir, const uint8_t* lut, int threads, int device, int batch, int shuffle,
                       uint64_t seed, int persistent, int nslots);
int ft_loader_next(void* l, int64_t* idx_out, void** dl_out);
int ft_loader_destroy(void* l);
}

using namespace pi05t;

namespace {
__global__ void gather_cams(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, uint8_t* out, int B) {
  const long long per = 224LL * 224 * 3, total = (long long)B * 3 * per;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const long long b = i / (3 * per), c = (i / per) % 3, o = i % per;
    const uint8_t* src = c == 0 ? c0 : (c == 1 ? c1 : c2);
    out[i] = src[b * per + o];
  }
}

bool save_state(const std::string& path, std::vector<TParam>& ps, int step, const pi05::WeightFile& cfg_src,
                const std::string& frozen_from) {
  // same container format as the reference / make_state files (Python Writer): text manifest + 256-byte aligned data
  struct T { std::string name; const float* dev; long long n; std::vector<int64_t> shape; bool host; };
  std::vector<T> ts;
  for (auto& P : ps) {
    ts.push_back({"p." + P.name, P.p, P.n, P.shape, false});
    if (P.ema) ts.push_back({"ema." + P.name, P.ema, P.n, P.shape, P.host_ema});
    ts.push_back({"adam_m." + P.name, P.m, P.n, P.shape, P.host_mv});
    ts.push_back({"adam_v." + P.name, P.v, P.n, P.shape, P.host_mv});
  }
  std::string man;
  for (auto& kv : cfg_src.cfg_all()) {
    if (kv.first == "step" || kv.first == "frozen_from") continue;
    man += "cfg " + kv.first;
    for (auto& v : kv.second) man += " " + v;
    man += "\n";
  }
  man += "cfg step " + std::to_string(step) + "\n";
  man += "cfg frozen_from " + frozen_from + "\n";  // frozen (f.) weights stay in the original state file
  uint64_t off = 0;
  std::vector<uint64_t> offs;
  for (auto& t : ts) {
    off = (off + 255) / 256 * 256;
    offs.push_back(off);
    man += "t " + t.name + " f32 " + std::to_string(off) + " " + std::to_string(t.n * 4) + " " +
           std::to_string(t.shape.size());
    for (auto d : t.shape) man += " " + std::to_string(d);
    man += "\n";
    off += t.n * 4;
  }
  const uint64_t mlen = man.size(), doff = (24 + mlen + 4095) / 4096 * 4096;
  std::ofstream f(path + ".partial", std::ios::binary);
  f.write("PI05W\0\0\1", 8);
  f.write((const char*)&mlen, 8);
  f.write((const char*)&doff, 8);
  f.write(man.data(), mlen);
  std::vector<char> pad(doff - 24 - mlen, 0);
  f.write(pad.data(), pad.size());
  uint64_t pos = 0;
  std::vector<float> h;
  for (size_t i = 0; i < ts.size(); ++i) {
    if (offs[i] > pos) {
      std::vector<char> z(offs[i] - pos, 0);
      f.write(z.data(), z.size());
    }
    h.resize(ts[i].n);
    PI05_CUDA(cudaMemcpy(h.data(), ts[i].dev, ts[i].n * 4, cudaMemcpyDefault));
    f.write((const char*)h.data(), ts[i].n * 4);
    pos = offs[i] + ts[i].n * 4;
  }
  f.close();
  return std::rename((path + ".partial").c_str(), path.c_str()) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  std::string state, model, table, lut_path, out = ".";
  int steps = 100, B = 0, offload = 2, save_every = 1000, log_every = 10, threads = 4;
  uint64_t seed = 42;
  bool aug = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--state") state = next();
    else if (a == "--model") model = next();
    else if (a == "--table") table = next();
    else if (a == "--lut") lut_path = next();
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--batch") B = std::stoi(next());
    else if (a == "--offload") offload = std::stoi(next());
    else if (a == "--seed") seed = std::stoull(next());
    else if (a == "--out") out = next();
    else if (a == "--save-every") save_every = std::stoi(next());
    else if (a == "--log-every") log_every = std::stoi(next());
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--no-aug") aug = false;
  }
  std::string err;
  pi05::WeightFile sf;
  if (!sf.open(state, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  if (B <= 0) B = sf.cfg_int("batch", 32);
  const int start = sf.cfg_int("step", 0);
  const bool lora = sf.cfg_str("mode", "expert") == "lora";
  std::unique_ptr<Trainer> te;
  std::unique_ptr<LoraTrainer> tl;
  bool ok;
  if (lora) { tl = std::make_unique<LoraTrainer>(); ok = tl->init(state, model, &err, offload); }
  else { te = std::make_unique<Trainer>(); ok = te->init(state, model, &err, offload); }
  if (!ok) { fprintf(stderr, "init: %s\n", err.c_str()); return 2; }
  std::vector<TParam>& ps = lora ? tl->params() : te->params();
  // resume: a state saved by this program carries Adam moments, EMA and the step (the optax count)
  if (lora) tl->set_step_count(start); else te->set_step_count(start);

  std::vector<uint8_t> lut((size_t)(1 << 24) * 3);
  {
    FILE* lf = fopen(lut_path.c_str(), "rb");
    if (!lf || fread(lut.data(), 1, lut.size(), lf) != lut.size()) { fprintf(stderr, "cannot read LUT %s\n", lut_path.c_str()); return 2; }
    fclose(lf);
  }
  void* loader = ft_loader_create(table.c_str(), lut.data(), threads, 0, B, 1, seed, 1, 4);
  if (!loader) { fprintf(stderr, "loader: %s\n", ft_last_error()); return 2; }

  uint8_t* imgs = nullptr;
  float* augd = nullptr;
  AugParams* dprm = nullptr;
  PI05_CUDA(cudaMalloc(&imgs, (size_t)B * 3 * 224 * 224 * 3));
  PI05_CUDA(cudaMalloc(&augd, (size_t)B * 3 * 224 * 224 * 3 * 4));
  PI05_CUDA(cudaMalloc(&dprm, sizeof(AugParams) * B * 3));
  const JaxKey train0 = jr_child(pi05::jax_key(seed), 0);  // scripts/train.py: train_rng, init_rng = split(key(seed))
  std::vector<int64_t> idx(B);
  printf("pi05_train: mode %s, %zu tensors, batch %d, steps %d..%d, offload %d, augmentation %s\n",
         lora ? "lora" : "expert", ps.size(), B, start, start + steps, offload, aug ? "on" : "off");
  double tsum = 0, lsum = 0;
  for (int step = start; step < start + steps; ++step) {
    auto t0 = std::chrono::steady_clock::now();
    void* dl[10];
    if (ft_loader_next(loader, idx.data(), dl)) { fprintf(stderr, "loader: %s\n", ft_last_error()); return 2; }
    auto* D = reinterpret_cast<DLManagedTensor**>(dl);
    const int T = (int)D[7]->dl_tensor.shape[1], H = (int)D[9]->dl_tensor.shape[1], M = (int)D[9]->dl_tensor.shape[2];
    std::vector<int32_t> tok((size_t)B * T);
    std::vector<uint8_t> tm((size_t)B * T), imask(3 * B);
    std::vector<float> act((size_t)B * H * M);
    gather_cams<<<2048, 256>>>((const uint8_t*)D[0]->dl_tensor.data, (const uint8_t*)D[1]->dl_tensor.data,
                               (const uint8_t*)D[2]->dl_tensor.data, imgs, B);
    PI05_CUDA(cudaMemcpy(tok.data(), D[7]->dl_tensor.data, tok.size() * 4, cudaMemcpyDeviceToHost));
    PI05_CUDA(cudaMemcpy(tm.data(), D[8]->dl_tensor.data, tm.size(), cudaMemcpyDeviceToHost));
    PI05_CUDA(cudaMemcpy(act.data(), D[9]->dl_tensor.data, act.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < 3; ++c)
      PI05_CUDA(cudaMemcpy(imask.data() + c * B, D[3 + c]->dl_tensor.data, B, cudaMemcpyDeviceToHost));
    for (int k = 0; k < 10; ++k) D[k]->deleter(D[k]);  // slot goes back to the loader
    for (uint8_t m : imask)
      if (!m) { fprintf(stderr, "a camera image is masked out: not supported by this trainer\n"); return 2; }
    if (H != 32 || M != 32) { fprintf(stderr, "unexpected action chunk %dx%d\n", H, M); return 2; }
    // openpi randomness of this step and the augmentation
    const StepRandom rnd = jr_step(train0, (uint32_t)step, B, 32 * 32);
    if (aug) {
      std::vector<AugParams> prm(B * 3);
      const std::vector<JaxKey> sk = jr_split(rnd.preprocess, B);
      for (int b = 0; b < B; ++b)
        for (int c = 0; c < 3; ++c) prm[b * 3 + c] = aug_params(sk[b], c > 0);
      PI05_CUDA(cudaMemcpy(dprm, prm.data(), sizeof(AugParams) * B * 3, cudaMemcpyHostToDevice));
    } else {
      std::vector<AugParams> prm(B * 3);  // identity: no geometry, jitter not applied
      PI05_CUDA(cudaMemcpy(dprm, prm.data(), sizeof(AugParams) * B * 3, cudaMemcpyHostToDevice));
    }
    augment(imgs, nullptr, dprm, augd, B, 0);
    PI05_CUDA(cudaDeviceSynchronize());
    if (lora) tl->zero_grads(); else te->zero_grads();
    double loss = 0;
    for (int b = 0; b < B; ++b) {
      std::vector<int> tokens;
      for (int i = 0; i < T; ++i)
        if (tm[(size_t)b * T + i]) tokens.push_back(tok[(size_t)b * T + i]);
      const float* img = augd + (size_t)b * 3 * 224 * 224 * 3;
      if (lora) {
        LoraSample s;
        s.img_f32_dev = img; s.tokens = tokens; s.actions = act.data() + (size_t)b * 1024;
        s.noise = rnd.noise.data() + (size_t)b * 1024; s.time = rnd.time[b];
        loss += tl->accumulate(s, 1.0f / B);
      } else {
        Sample s;
        s.img_f32_dev = img; s.tokens = tokens; s.actions = act.data() + (size_t)b * 1024;
        s.noise = rnd.noise.data() + (size_t)b * 1024; s.time = rnd.time[b];
        loss += te->accumulate(s, 1.0f / B);
      }
    }
    double gn;
    if (lora) { tl->finalize_grads(); gn = tl->grad_norm(); tl->opt_step(); }
    else { te->finalize_grads(); gn = te->grad_norm(); te->opt_step(); }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    tsum += ms;
    lsum += loss / B;
    if ((step + 1 - start) % log_every == 0) {
      printf("step %d  loss %.5f  grad_norm %.4f  %.0f ms/step\n", step + 1, lsum / log_every, gn, tsum / log_every);
      fflush(stdout);
      tsum = lsum = 0;
    }
    if ((step + 1) % save_every == 0 || step + 1 == start + steps) {
      const std::string p = out + "/state_step" + std::to_string(step + 1) + ".pi05d";
      printf("saving %s: %s\n", p.c_str(), save_state(p, ps, step + 1, sf, sf.cfg_str("frozen_from", state)) ? "ok" : "FAILED");
    }
  }
  ft_loader_destroy(loader);
  return 0;
}
