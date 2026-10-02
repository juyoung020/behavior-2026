// Training-step randomness and augmentation vs JAX (tools/aug_ref.py): key chain (bit-exact), noise, flow time,
// augmented images. Floor = the JAX dump on the other platform.
//   pi05_aug_test --ref DIR --tag gpu [--floor cpu]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../pi05_native/src/weights.h"
#include "../src/augment.cuh"

using namespace pi05t;

static std::vector<float> rd(const pi05::WeightFile& f, const std::string& n) {
  std::vector<float> v;
  const pi05::TensorInfo* t = f.find(n);
  if (!t) return v;
  std::string e;
  v.resize(t->numel());
  f.read(*t, v.data(), &e);
  return v;
}
static void cmp(const char* what, const std::vector<float>& a, const std::vector<float>& r, const std::vector<float>& fl) {
  double d2 = 0, r2 = 0, mx = 0, f2 = 0, fmx = 0;
  size_t eq = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    const double d = (double)a[i] - r[i];
    d2 += d * d;
    r2 += (double)r[i] * r[i];
    mx = std::max(mx, std::fabs(d));
    eq += a[i] == r[i];
    if (!fl.empty()) {
      const double e = (double)fl[i] - r[i];
      f2 += e * e;
      fmx = std::max(fmx, std::fabs(e));
    }
  }
  printf("%-26s rel %.3e max %.3e bit-equal %.4f   floor rel %.3e max %.3e\n", what, std::sqrt(d2 / (r2 + 1e-30)), mx,
         (double)eq / r.size(), std::sqrt(f2 / (r2 + 1e-30)), fmx);
}

int main(int argc, char** argv) {
  std::string ref, tag = "gpu", floor_tag;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ref") ref = argv[++i];
    else if (a == "--tag") tag = argv[++i];
    else if (a == "--floor") floor_tag = argv[++i];
  }
  std::string err;
  pi05::WeightFile R, F;
  if (!R.open(ref + "/aug_" + tag + ".pi05d", &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  const bool hf = !floor_tag.empty() && F.open(ref + "/aug_" + floor_tag + ".pi05d", &err);
  const int B = R.cfg_int("batch", 4), seed = R.cfg_int("seed", 42);
  std::vector<uint8_t> img(R.find("in.img")->nbytes);
  R.read(*R.find("in.img"), img.data(), &err);
  uint8_t* dimg;
  float* dout;
  AugParams* dp;
  cudaMalloc(&dimg, img.size());
  cudaMalloc(&dout, img.size() * 4);
  cudaMalloc(&dp, sizeof(AugParams) * B * 3);
  cudaMemcpy(dimg, img.data(), img.size(), cudaMemcpyHostToDevice);
  const JaxKey train0 = jr_child(pi05::jax_key((uint64_t)seed), 0);  // train_rng, _ = split(key(seed))
  std::string steps = R.cfg_str("steps", "0");
  int fails = 0;
  for (size_t p = 0; p <= steps.size();) {
    size_t q = steps.find(',', p);
    if (q == std::string::npos) q = steps.size();
    const int s = std::stoi(steps.substr(p, q - p));
    p = q + 1;
    const std::string pre = "s" + std::to_string(s) + ".";
    const JaxKey r = jr_fold_in(train0, (uint32_t)s);
    std::vector<int32_t> kd(2);
    R.read(*R.find(pre + "key"), kd.data(), &err);
    const bool key_ok = (uint32_t)kd[0] == r.k0 && (uint32_t)kd[1] == r.k1;
    printf("step %d: train key %s\n", s, key_ok ? "bit-identical" : "DIFFERENT");
    fails += !key_ok;
    const StepRandom sr = jr_step(train0, (uint32_t)s, B, 32 * 32);
    cmp("  noise", sr.noise, rd(R, pre + "noise"), hf ? rd(F, pre + "noise") : std::vector<float>{});
    cmp("  flow time", sr.time, rd(R, pre + "time"), hf ? rd(F, pre + "time") : std::vector<float>{});
    std::vector<AugParams> prm(B * 3);
    const std::vector<JaxKey> sk = jr_split(sr.preprocess, B);  // model.py:183 split(rng, B), same for every camera
    for (int b = 0; b < B; ++b)
      for (int c = 0; c < 3; ++c) prm[b * 3 + c] = aug_params(sk[b], c > 0);
    cudaMemcpy(dp, prm.data(), sizeof(AugParams) * B * 3, cudaMemcpyHostToDevice);
    augment(dimg, nullptr, dp, dout, B, 0);
    std::vector<float> out(img.size());
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    cmp("  augmented images", out, rd(R, pre + "img"), hf ? rd(F, pre + "img") : std::vector<float>{});
    for (int b = 0; b < B; ++b)
      printf("    sample %d: crop (%.3f, %.3f) jitter apply %d/%d/%d\n", b, prm[b * 3].cy, prm[b * 3].cx, prm[b * 3].apply,
             prm[b * 3 + 1].apply, prm[b * 3 + 2].apply);
  }
  printf("%s\n", fails ? "FAIL (key chain)" : "key chain ok");
  return fails;
}
