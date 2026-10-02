// pi05_infer_batch / pi05_act_batch: bit identity with the single-episode API, and latency / throughput / memory.
//   pi05_batch_test --weights W.pi05w --ref DIR [--cap 64] [--sizes 1,16,64,256] [--act-steps 40]
// DIR: inference reference dumps (tools/dump_reference*.py) providing in.img, in.proprio, in.prompt / in.task_stage,
// in.noise (and inp.kept for PiBehavior inpainting).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../include/pi05_batch.h"
#include "../src/weights.h"

using namespace pi05;

static double gib_used() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return (double)(t - f) / (1 << 30);
}

struct Sample {
  std::vector<uint8_t> img;
  std::vector<float> proprio, noise;
  std::string prompt;
  int task = 0, stage = 0;
  std::vector<double> kept;  // PiBehavior [4][23]
};

int main(int argc, char** argv) {
  std::string weights, ref, sizes = "1,16,64,256";
  int cap = 64, act_steps = 40;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--weights") weights = next();
    else if (a == "--ref") ref = next();
    else if (a == "--cap") cap = std::stoi(next());
    else if (a == "--sizes") sizes = next();
    else if (a == "--act-steps") act_steps = std::stoi(next());
  }
  const double base = gib_used();
  char err[512];
  auto t0 = std::chrono::steady_clock::now();
  Pi05Engine* e = pi05_create_batch(weights.c_str(), 0, cap, err, sizeof err);
  if (!e) { fprintf(stderr, "create: %s\n", err); return 2; }
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  Pi05Info info;
  pi05_info(e, &info);
  const bool pb = info.model_kind == 1;
  const double mem = gib_used();
  printf("%s engine, batch capacity %d: load %.1f s, GPU %.2f GiB used by the engine (weights %.2f GiB)\n",
         pb ? "PiBehavior" : "pi05", cap, load_s, mem - base, info.weight_bytes / 1073741824.0);
  // samples
  std::vector<Sample> S;
  for (int k = 0; k < 4; ++k) {
    WeightFile f;
    std::string m;
    char nm[64];
    snprintf(nm, sizeof nm, "/gpu_%03d.pi05d", k);
    if (!f.open(ref + nm, &m)) { fprintf(stderr, "%s\n", m.c_str()); return 2; }
    Sample s;
    s.img.resize(f.find("in.img")->nbytes);
    f.read(*f.find("in.img"), s.img.data(), &m);
    s.proprio.resize(f.find("in.proprio")->numel());
    f.read(*f.find("in.proprio"), s.proprio.data(), &m);
    s.noise.resize(f.find("in.noise")->numel());
    f.read(*f.find("in.noise"), s.noise.data(), &m);
    if (pb) {
      int ts[2];
      f.read(*f.find("in.task_stage"), ts, &m);
      s.task = ts[0];
      s.stage = ts[1];
      if (const TensorInfo* t = f.find("inp.kept")) {
        s.kept.resize(t->numel());
        f.read(*t, s.kept.data(), &m);
      }
    } else {
      std::vector<uint8_t> p(f.find("in.prompt")->nbytes);
      f.read(*f.find("in.prompt"), p.data(), &m);
      s.prompt.assign(p.begin(), p.end());
    }
    S.push_back(std::move(s));
  }
  const int ah = info.action_horizon, na = info.action_dim, np = (int)S[0].proprio.size();
  auto views_of = [&](const Sample& s, Pi05View* v) {
    for (int c = 0; c < 3; ++c) v[c] = {s.img.data() + (size_t)c * 224 * 224 * 3, 224, 224, 224 * 3, 3, 0};
  };
  auto single = [&](const Sample& s, bool inpaint, bool given_noise, double* out, float* logits) {
    Pi05Image im[3];
    for (int c = 0; c < 3; ++c) im[c] = {s.img.data() + (size_t)c * 224 * 224 * 3, 224, 224, 224 * 3, 3, 0};
    const float* nz = given_noise ? s.noise.data() : nullptr;
    if (pb)
      return pi05_infer_pb(e, im, s.proprio.data(), np, s.task, s.stage, inpaint ? s.kept.data() : nullptr,
                           inpaint ? 4 : 0, nz, out, logits, nullptr);
    return pi05_infer(e, im, s.proprio.data(), np, s.prompt.c_str(), nz, out, nullptr);
  };
  int fails = 0;
  // ---- 1) chunk API: every episode vs its single inference (given noise), several batch sizes
  std::vector<std::vector<double>> ref_a(4, std::vector<double>((size_t)ah * na));
  std::vector<std::vector<float>> ref_l(4, std::vector<float>(15));
  for (int k = 0; k < 4; ++k)
    if (single(S[k], false, true, ref_a[k].data(), ref_l[k].data())) { fprintf(stderr, "%s\n", pi05_last_error(e)); return 2; }
  printf("\n%6s %10s %12s %14s %8s   %s\n", "n", "total ms", "ms/episode", "episodes/s", "chunks", "bit-identical to single");
  size_t p = 0;
  while (p <= sizes.size()) {
    size_t q = sizes.find(',', p);
    if (q == std::string::npos) q = sizes.size();
    const int n = std::stoi(sizes.substr(p, q - p));
    p = q + 1;
    std::vector<Pi05View> v((size_t)n * 3);
    std::vector<float> prop((size_t)n * np), noise((size_t)n * ah * 32);
    std::vector<const char*> prompts(n);
    std::vector<int32_t> task(n), stage(n);
    for (int i = 0; i < n; ++i) {
      const Sample& s = S[i % 4];
      views_of(s, &v[(size_t)i * 3]);
      memcpy(&prop[(size_t)i * np], s.proprio.data(), np * 4);
      memcpy(&noise[(size_t)i * ah * 32], s.noise.data(), (size_t)ah * 32 * 4);
      prompts[i] = s.prompt.c_str();
      task[i] = s.task;
      stage[i] = s.stage;
    }
    Pi05BatchIn in{};
    in.n = n; in.views = v.data(); in.proprio = prop.data(); in.n_proprio = np; in.prompts = prompts.data();
    in.task = task.data(); in.stage = stage.data(); in.noise = noise.data();
    std::vector<double> acts((size_t)n * ah * na);
    std::vector<float> lg((size_t)n * 15);
    Pi05BatchOut out{acts.data(), lg.data()};
    Pi05BatchTiming t{};
    pi05_infer_batch(e, &in, &out, &t);  // warm-up (captures, plans)
    const int reps = n <= 16 ? 3 : 1;
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
      if (pi05_infer_batch(e, &in, &out, &t)) { fprintf(stderr, "%s\n", pi05_last_error(e)); return 2; }
      best = std::min(best, (double)t.total_ms);
    }
    int same = 0;
    for (int i = 0; i < n; ++i) {
      const bool a_ok = memcmp(&acts[(size_t)i * ah * na], ref_a[i % 4].data(), (size_t)ah * na * 8) == 0;
      const bool l_ok = !pb || memcmp(&lg[(size_t)i * 15], ref_l[i % 4].data(), 15 * 4) == 0;
      same += a_ok && l_ok;
    }
    fails += same != n;
    printf("%6d %10.1f %12.2f %14.1f %8d   %d/%d  (prefix %.0f ms, suffix %.0f ms, host %.0f ms)\n", n, best, best / n,
           1000.0 * n / best, t.chunks, same, n, t.prefix_ms, t.suffix_ms, t.host_ms);
  }
  printf("GPU used by the engine after the runs: %.2f GiB\n", gib_used() - base);
  // ---- 2) engine random stream (+ PiBehavior inpainting): batch vs sequential single calls in the same order
  {
    const int n = std::min(cap, 12);
    std::vector<double> seq((size_t)n * ah * na), bat((size_t)n * ah * na);
    pi05_seed(e, 7);
    for (int i = 0; i < n; ++i) single(S[i % 4], pb && (i & 1) && !S[i % 4].kept.empty(), false, &seq[(size_t)i * ah * na], nullptr);
    std::vector<Pi05View> v((size_t)n * 3);
    std::vector<float> prop((size_t)n * np);
    std::vector<const char*> prompts(n);
    std::vector<int32_t> task(n), stage(n);
    std::vector<double> init((size_t)n * 4 * 23, 0.0);
    std::vector<uint8_t> use(n, 0);
    for (int i = 0; i < n; ++i) {
      const Sample& s = S[i % 4];
      views_of(s, &v[(size_t)i * 3]);
      memcpy(&prop[(size_t)i * np], s.proprio.data(), np * 4);
      prompts[i] = s.prompt.c_str();
      task[i] = s.task;
      stage[i] = s.stage;
      if (pb && (i & 1) && !s.kept.empty()) {
        memcpy(&init[(size_t)i * 92], s.kept.data(), 92 * 8);
        use[i] = 1;
      }
    }
    Pi05BatchIn in{};
    in.n = n; in.views = v.data(); in.proprio = prop.data(); in.n_proprio = np; in.prompts = prompts.data();
    in.task = task.data(); in.stage = stage.data();
    if (pb) { in.initial_actions = init.data(); in.use_initial = use.data(); }
    Pi05BatchOut out{bat.data(), nullptr};
    pi05_seed(e, 7);
    if (pi05_infer_batch(e, &in, &out, nullptr)) { fprintf(stderr, "%s\n", pi05_last_error(e)); return 2; }
    const bool ok = memcmp(seq.data(), bat.data(), seq.size() * 8) == 0;
    fails += !ok;
    printf("\nengine noise stream%s, %d episodes: batch %s sequential calls\n", pb ? " + inpainting on odd episodes" : "",
           n, ok ? "bit-identical to" : "DIFFERS from");
  }
  // ---- 3) wrappers: pi05_act_batch vs pi05_act slot by slot, act_steps environment steps
  {
    const int n = std::min(cap, 8);
    std::vector<float> seq((size_t)act_steps * n * na), bat((size_t)act_steps * n * na);
    auto setup = [&]() {
      pi05_reset(e, -1);
      pi05_seed(e, 11);
      if (pb) for (int i = 0; i < n; ++i) pi05_set_task(e, i, S[i % 4].task);
    };
    setup();
    for (int t = 0; t < act_steps; ++t)
      for (int i = 0; i < n; ++i) {
        const Sample& s = S[(i + t / 7) % 4];  // observations change over time
        Pi05Image im[3];
        for (int c = 0; c < 3; ++c) im[c] = {s.img.data() + (size_t)c * 224 * 224 * 3, 224, 224, 224 * 3, 3, 0};
        if (pi05_act(e, i, im, s.proprio.data(), np, s.prompt.c_str(), 16, &seq[((size_t)t * n + i) * na], nullptr) < 0) {
          fprintf(stderr, "%s\n", pi05_last_error(e));
          return 2;
        }
      }
    setup();
    int inferred = 0;
    for (int t = 0; t < act_steps; ++t) {
      std::vector<Pi05View> v((size_t)n * 3);
      std::vector<float> prop((size_t)n * np);
      std::vector<const char*> prompts(n);
      std::vector<int32_t> slots(n), task(n);
      for (int i = 0; i < n; ++i) {
        const Sample& s = S[(i + t / 7) % 4];
        views_of(s, &v[(size_t)i * 3]);
        memcpy(&prop[(size_t)i * np], s.proprio.data(), np * 4);
        prompts[i] = s.prompt.c_str();
        slots[i] = i;
        task[i] = S[i % 4].task;
      }
      Pi05ActBatchIn in{};
      in.n = n; in.slots = slots.data(); in.views = v.data(); in.proprio = prop.data(); in.n_proprio = np;
      in.prompts = prompts.data(); in.replan_every = 16; in.task = pb ? task.data() : nullptr;
      std::vector<uint8_t> nc(n);
      if (pi05_act_batch(e, &in, &bat[(size_t)t * n * na], nc.data(), nullptr)) {
        fprintf(stderr, "%s\n", pi05_last_error(e));
        return 2;
      }
      for (uint8_t x : nc) inferred += x;
    }
    const bool ok = memcmp(seq.data(), bat.data(), seq.size() * 4) == 0;
    fails += !ok;
    printf("wrappers: %d slots x %d steps (%d inferences): pi05_act_batch %s pi05_act\n", n, act_steps, inferred,
           ok ? "bit-identical to" : "DIFFERS from");
  }
  pi05_destroy(e);
  printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
