// One-step memory / time measurement of the native trainer at full size (no reference comparison).
//   pi05_train_bench --state S.pi05d --model pi05_radio.pi05w --ref DIR(inference dumps) [--batch 32] [--steps 3]
//                    [--offload 0|1|2]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <memory>

#include "../src/lora.h"
#include "../src/trainer.h"

using namespace pi05t;

static double gpu_used_gib() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return (double)(t - f) / (1 << 30);
}

int main(int argc, char** argv) {
  std::string state, model, ref;
  int B = 32, steps = 3, offload = 2;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--state") state = next();
    else if (a == "--model") model = next();
    else if (a == "--ref") ref = next();
    else if (a == "--batch") B = std::stoi(next());
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--offload") offload = std::stoi(next());
  }
  const double base = gpu_used_gib();
  std::string err;
  pi05::WeightFile sf;
  if (!sf.open(state, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 2;
  }
  const bool lora = sf.cfg_str("mode", "expert") == "lora";
  std::unique_ptr<Trainer> te;
  std::unique_ptr<LoraTrainer> tl;
  bool ok;
  if (lora) {
    tl = std::make_unique<LoraTrainer>();
    ok = tl->init(state, model, &err, offload);
  } else {
    te = std::make_unique<Trainer>();
    ok = te->init(state, model, &err, offload);
  }
  if (!ok) {
    fprintf(stderr, "init: %s\n", err.c_str());
    return 2;
  }
  long long np = 0;
  for (auto& P : lora ? tl->params() : te->params()) np += P.n;
  const double after_init = gpu_used_gib();
  // inputs: the first 4 radio demo frames of the inference reference, cycled
  struct In { std::vector<uint8_t> img; std::vector<int> tok; std::vector<float> act, noise; };
  std::vector<In> ins;
  for (int k = 0; k < 4; ++k) {
    pi05::WeightFile f;
    char nm[64];
    snprintf(nm, sizeof nm, "/gpu_%03d.pi05d", k);
    if (!f.open(ref + nm, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    In in;
    in.img.resize(f.find("in.img")->nbytes);
    f.read(*f.find("in.img"), in.img.data(), &err);
    std::vector<int32_t> t(200);
    std::vector<uint8_t> m(200);
    f.read(*f.find("host.tokens"), t.data(), &err);
    f.read(*f.find("host.mask"), m.data(), &err);
    for (int i = 0; i < 200; ++i)
      if (m[i]) in.tok.push_back(t[i]);
    in.act.resize(32 * 32);
    in.noise.resize(32 * 32);
    f.read(*f.find("out.actions_raw"), in.act.data(), &err);
    f.read(*f.find("in.noise"), in.noise.data(), &err);
    ins.push_back(std::move(in));
  }
  double peak = after_init;
  for (int s = 0; s < steps; ++s) {
    auto t0 = std::chrono::steady_clock::now();
    if (lora) tl->zero_grads(); else te->zero_grads();
    double loss = 0;
    for (int b = 0; b < B; ++b) {
      const In& in = ins[b % 4];
      const float tm = 0.1f + 0.8f * (float)((b * 7 + s) % 32) / 31.0f;
      if (lora) {
        LoraSample smp;
        smp.img_u8 = in.img.data(); smp.tokens = in.tok; smp.actions = in.act.data(); smp.noise = in.noise.data();
        smp.time = tm;
        loss += tl->accumulate(smp, 1.0f / B);
      } else {
        Sample smp;
        smp.img = in.img.data(); smp.tokens = in.tok; smp.actions = in.act.data(); smp.noise = in.noise.data();
        smp.time = tm;
        loss += te->accumulate(smp, 1.0f / B);
      }
      peak = std::max(peak, gpu_used_gib());
    }
    if (lora) tl->finalize_grads(); else te->finalize_grads();
    auto t1 = std::chrono::steady_clock::now();
    if (lora) tl->opt_step(); else te->opt_step();
    auto t2 = std::chrono::steady_clock::now();
    const double fb = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double op = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("step %d: loss %.4f  forward+backward %.0f ms (%.1f ms/sample)  optimizer %.0f ms  step %.0f ms\n", s + 1,
           loss / B, fb, fb / B, op, fb + op);
  }
  printf("trainable %.1f M params, offload %d; GPU used: before %.2f GiB, after init %.2f GiB, peak %.2f GiB "
         "(whole device, other processes included)\n",
         np / 1e6, offload, base, after_init, peak);
  return 0;
}
