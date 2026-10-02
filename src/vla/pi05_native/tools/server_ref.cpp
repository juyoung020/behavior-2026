// In-process reference for the submission server check: the observation sequence written by
// server_protocol_test.py through pi05_act_batch on a fresh engine (same weights, default seed, same prompt as the
// server default), compared bit for bit with the actions the server returned to the official client.
//   pi05_server_ref --weights W.pi05w --seq S.pi05d --actions A.f32 [--prompt "..."]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../include/pi05_batch.h"
#include "../src/weights.h"

int main(int argc, char** argv) {
  std::string weights, seq, acts, prompt = "Turn on the radio receiver that's on the table in the living room.";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--weights") weights = next();
    else if (a == "--seq") seq = next();
    else if (a == "--actions") acts = next();
    else if (a == "--prompt") prompt = next();
  }
  std::string err;
  pi05::WeightFile S;
  if (!S.open(seq, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  const int E = S.cfg_int("envs", 2), T = S.cfg_int("steps", 60);
  char e[512];
  Pi05Engine* eng = pi05_create_batch(weights.c_str(), 0, 16, e, sizeof e);
  if (!eng) { fprintf(stderr, "%s\n", e); return 2; }
  Pi05Info info;
  pi05_info(eng, &info);
  const int ad = info.action_dim;
  std::vector<float> ref((size_t)T * E * ad), got((size_t)T * E * ad);
  FILE* f = fopen(acts.c_str(), "rb");
  if (!f || fread(got.data(), 4, got.size(), f) != got.size()) { fprintf(stderr, "cannot read %s\n", acts.c_str()); return 2; }
  fclose(f);
  std::vector<int32_t> slots(E);
  for (int i = 0; i < E; ++i) slots[i] = i;
  for (int t = 0; t < T; ++t) {
    const std::string p = "s" + std::to_string(t) + ".";
    std::vector<std::vector<uint8_t>> cam(3);
    std::vector<Pi05View> views((size_t)E * 3);
    for (int c = 0; c < 3; ++c) {
      const pi05::TensorInfo* ti = S.find(p + "cam" + std::to_string(c));
      cam[c].resize(ti->nbytes);
      S.read(*ti, cam[c].data(), &err);
      const int H = (int)ti->shape[1], W = (int)ti->shape[2], C = (int)ti->shape[3];
      for (int b = 0; b < E; ++b)
        views[(size_t)b * 3 + c] = {cam[c].data() + (size_t)b * H * W * C, H, W, (int64_t)W * C, C, 0};
    }
    const pi05::TensorInfo* tp = S.find(p + "proprio");
    std::vector<float> prop(tp->numel());
    S.read(*tp, prop.data(), &err);
    std::vector<int32_t> task(E);
    S.read(*S.find(p + "task"), task.data(), &err);
    Pi05ActBatchIn in{};
    in.n = E; in.slots = slots.data(); in.views = views.data(); in.proprio = prop.data();
    in.n_proprio = (int)tp->shape[1]; in.prompt = prompt.c_str(); in.replan_every = 16;
    in.task = info.model_kind == 1 ? task.data() : nullptr;
    if (pi05_act_batch(eng, &in, &ref[(size_t)t * E * ad], nullptr, nullptr)) {
      fprintf(stderr, "%s\n", pi05_last_error(eng));
      return 2;
    }
  }
  size_t eq = 0;
  double mx = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    eq += memcmp(&ref[i], &got[i], 4) == 0;
    mx = std::max(mx, (double)std::fabs(ref[i] - got[i]));
  }
  printf("server (official client) vs in-process pi05_act_batch: %zu/%zu values bit-identical, max |diff| %.3g -> %s\n",
         eq, ref.size(), mx, eq == ref.size() ? "PASS" : "FAIL");
  pi05_destroy(eng);
  return eq == ref.size() ? 0 : 1;
}
