// Native trainer vs openpi JAX training (tools/train_ref.py dumps): loss, every parameter gradient, parameters and EMA
// after optimizer steps 1 and N. Floor = the same JAX computation on the other platform (CPU vs GPU); pass if
// rel_ours <= 2 * rel_floor for every point (rel = ||ours - ref|| / ||ref||, parameters compared as updates p - p0).
//   pi05_train_verify --ref DIR --tag expert_gpu --floor expert_cpu [--offload 0|1|2]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <functional>
#include <memory>

#include "../src/lora.h"
#include "../src/trainer.h"

using namespace pi05t;
using pi05::WeightFile;

static std::vector<float> read_f32(const WeightFile& f, const std::string& n) {
  const pi05::TensorInfo* t = f.find(n);
  std::vector<float> v;
  if (!t) return v;
  std::string err;
  if (t->dtype == pi05::DType::F32) {
    v.resize(t->numel());
    f.read(*t, v.data(), &err);
  }
  return v;
}

struct Stat { double rel = 0, maxabs = 0; size_t imax = 0; };
static Stat cmp(const float* a, const float* b, size_t n, const float* base = nullptr) {
  double d2 = 0, r2 = 0, mx = 0;
  size_t im = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = a[i] - (base ? base[i] : 0), y = b[i] - (base ? base[i] : 0);
    d2 += (x - y) * (x - y);
    r2 += y * y;
    if (std::fabs(x - y) > mx) { mx = std::fabs(x - y); im = i; }
  }
  return {std::sqrt(d2 / (r2 + 1e-300)), mx, im};
}

int main(int argc, char** argv) {
  std::string ref, tag = "expert_gpu", floor_tag;
  int offload = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--ref") ref = next();
    else if (a == "--tag") tag = next();
    else if (a == "--floor") floor_tag = next();
    else if (a == "--offload") offload = std::stoi(next());
  }
  std::string err;
  WeightFile S, Gr, O, FG, FO;
  if (!S.open(ref + "/" + tag + "_state.pi05d", &err) || !Gr.open(ref + "/" + tag + "_grad.pi05d", &err) ||
      !O.open(ref + "/" + tag + "_opt.pi05d", &err)) {
    fprintf(stderr, "open: %s\n", err.c_str());
    return 2;
  }
  const bool hf = !floor_tag.empty() && FG.open(ref + "/" + floor_tag + "_grad.pi05d", &err) &&
                  FO.open(ref + "/" + floor_tag + "_opt.pi05d", &err);
  const int B = S.cfg_int("batch", 4), steps = S.cfg_int("steps", 10);
  const std::string model = ref + "/model_i" + std::to_string(S.cfg_int("img.depth", 2)) + "_l" +
                            std::to_string(S.cfg_int("llm.depth", 2)) + ".pi05w";
  // batch
  std::vector<uint8_t> img(S.find("in.img")->nbytes);
  S.read(*S.find("in.img"), img.data(), &err);
  std::vector<int32_t> tok(S.find("in.tokens")->numel());
  S.read(*S.find("in.tokens"), tok.data(), &err);
  std::vector<uint8_t> msk(S.find("in.mask")->numel());
  S.read(*S.find("in.mask"), msk.data(), &err);
  std::vector<float> act = read_f32(S, "in.actions"), noise = read_f32(S, "in.noise"), ftime = read_f32(S, "in.time");
  const int T = (int)(tok.size() / B);
  auto fill = [&](auto& s, int b, int step) {
    for (int i = 0; i < T; ++i)
      if (msk[(size_t)b * T + i]) s.tokens.push_back(tok[(size_t)b * T + i]);
    s.actions = act.data() + (size_t)b * 32 * 32;
    s.noise = noise.data() + ((size_t)step * B + b) * 32 * 32;
    s.time = ftime[(size_t)step * B + b];
  };
  const uint8_t* imgb = img.data();
  // both trainers behind one interface (mode from the state file)
  struct Iface {
    std::function<void()> zero, finalize, step;
    std::function<float(int, int)> acc;
    std::function<double()> gn;
    std::vector<TParam>* params = nullptr;
  } I;
  std::unique_ptr<Trainer> te;
  std::unique_ptr<LoraTrainer> tl;
  const std::string state = ref + "/" + tag + "_state.pi05d";
  if (S.cfg_str("mode", "expert") == "lora") {
    tl = std::make_unique<LoraTrainer>();
    if (!tl->init(state, model, &err, offload)) {
      fprintf(stderr, "init: %s\n", err.c_str());
      return 2;
    }
    LoraTrainer* t = tl.get();
    I = {[t] { t->zero_grads(); }, [t] { t->finalize_grads(); }, [t] { t->opt_step(); },
         [&, t](int b, int s) { LoraSample x; x.img_u8 = imgb + (size_t)b * 3 * 224 * 224 * 3; fill(x, b, s); return t->accumulate(x, 1.0f / B); },
         [t] { return t->grad_norm(); }, &t->params()};
  } else {
    te = std::make_unique<Trainer>();
    if (!te->init(state, model, &err, offload)) {
      fprintf(stderr, "init: %s\n", err.c_str());
      return 2;
    }
    Trainer* t = te.get();
    I = {[t] { t->zero_grads(); }, [t] { t->finalize_grads(); }, [t] { t->opt_step(); },
         [&, t](int b, int s) { Sample x; x.img = imgb + (size_t)b * 3 * 224 * 224 * 3; fill(x, b, s); return t->accumulate(x, 1.0f / B); },
         [t] { return t->grad_norm(); }, &t->params()};
  }
  printf("trainer: %zu trainable tensors, batch %d, steps %d, floor %s\n", (*I.params).size(), B, steps,
         hf ? floor_tag.c_str() : "none");

  struct Row { std::string name; Stat ours, fl; bool hf; double vo = 0, vr = 0, vf = 0, vb = 0; };
  std::vector<Row> rows;
  auto add = [&](const std::string& nm, const float* ours, const std::vector<float>& ref_v,
                 const std::vector<float>& fl_v, const float* base = nullptr) {
    if (ref_v.empty()) { printf("missing reference %s\n", nm.c_str()); return; }
    Row r{nm, cmp(ours, ref_v.data(), ref_v.size(), base), {}, !fl_v.empty()};
    if (r.hf) r.fl = cmp(fl_v.data(), ref_v.data(), ref_v.size(), base);
    const size_t i = r.ours.imax;
    const double b0 = base ? base[i] : 0;
    r.vo = ours[i] - b0; r.vr = ref_v[i] - b0; r.vf = r.hf ? fl_v[i] - b0 : NAN; r.vb = b0;
    rows.push_back(r);
  };

  // ---- gradient at the initial parameters
  auto t0 = std::chrono::steady_clock::now();
  I.zero();
  std::vector<float> lps;
  double lsum = 0;
  for (int b = 0; b < B; ++b) {
    const float l = I.acc(b, 0);
    lsum += l;
    lps.push_back(l);
  }
  I.finalize();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const float loss = (float)(lsum / B);
  {
    std::vector<float> ours{loss};
    add("loss", ours.data(), read_f32(Gr, "loss"), hf ? read_f32(FG, "loss") : std::vector<float>{});
    std::vector<float> rp = read_f32(Gr, "loss_ps"), fp = hf ? read_f32(FG, "loss_ps") : std::vector<float>{};
    auto rowmean = [&](const std::vector<float>& v) {
      std::vector<float> o;
      for (size_t i = 0; i + 32 <= v.size(); i += 32) {
        double s = 0;
        for (int j = 0; j < 32; ++j) s += v[i + j];
        o.push_back((float)(s / 32));
      }
      return o;
    };
    add("loss per sample", lps.data(), rowmean(rp), rowmean(fp));
  }
  const double gn = I.gn();
  {
    std::vector<float> o{(float)gn};
    add("grad global norm", o.data(), read_f32(Gr, "grad_norm"), hf ? read_f32(FG, "grad_norm") : std::vector<float>{});
  }
  std::vector<std::vector<float>> p0((*I.params).size());
  for (size_t i = 0; i < (*I.params).size(); ++i) {
    const TParam& P = (*I.params)[i];
    std::vector<float> g(P.n);
    PI05_CUDA(cudaMemcpy(g.data(), P.g, P.n * 4, cudaMemcpyDeviceToHost));
    const std::vector<float> rg = read_f32(Gr, "g." + P.name), fg = hf ? read_f32(FG, "g." + P.name) : std::vector<float>{};
    add("grad " + P.name, g.data(), rg, fg);
    // stacked layers (leading depth axis): also per layer, to localise a mismatch
    const bool stacked = P.name.find("/layers/") != std::string::npos || P.name.find("/encoderblock/") != std::string::npos;
    if (stacked && getenv("PI05_VERIFY_LAYERS") && !rg.empty() && P.shape.size() > 1) {
      const size_t per = P.n / P.shape[0];
      for (int64_t l = 0; l < P.shape[0]; ++l) {
        std::vector<float> a(rg.begin() + l * per, rg.begin() + (l + 1) * per), b;
        if (!fg.empty()) b.assign(fg.begin() + l * per, fg.begin() + (l + 1) * per);
        add("  [" + std::to_string(l) + "] " + P.name, g.data() + l * per, a, b);
      }
    }
    p0[i].resize(P.n);
    PI05_CUDA(cudaMemcpy(p0[i].data(), P.p, P.n * 4, cudaMemcpyDeviceToHost));
  }
  // ---- optimizer steps (step s uses noise[s], time[s])
  std::vector<float> step_loss{loss};
  for (int s = 0; s < steps; ++s) {
    if (s > 0) {
      I.zero();
      double ls = 0;
      for (int b = 0; b < B; ++b) ls += I.acc(b, s);
      I.finalize();
      step_loss.push_back((float)(ls / B));
    }
    I.step();
    if (s + 1 == 1 || s + 1 == steps) {
      const std::string pre = "s" + std::to_string(s + 1) + ".";
      for (size_t i = 0; i < (*I.params).size(); ++i) {
        const TParam& P = (*I.params)[i];
        std::vector<float> p(P.n);
        PI05_CUDA(cudaMemcpy(p.data(), P.p, P.n * 4, cudaMemcpyDeviceToHost));
        add("step " + std::to_string(s + 1) + " update " + P.name, p.data(), read_f32(O, pre + "p." + P.name),
            hf ? read_f32(FO, pre + "p." + P.name) : std::vector<float>{}, p0[i].data());
        if (P.ema) {
          PI05_CUDA(cudaMemcpy(p.data(), P.ema, P.n * 4, cudaMemcpyDefault));
          add("step " + std::to_string(s + 1) + " ema-p0 " + P.name, p.data(), read_f32(O, pre + "ema." + P.name),
              hf ? read_f32(FO, pre + "ema." + P.name) : std::vector<float>{}, p0[i].data());
        }
      }
    }
  }
  add("loss per step", step_loss.data(), read_f32(O, "loss_steps"), hf ? read_f32(FO, "loss_steps") : std::vector<float>{});

  printf("\n%-86s %10s %10s %10s %10s %5s\n", "point", "rel_ours", "rel_floor", "max_ours", "max_floor", "ok");
  int fails = 0;
  for (auto& r : rows) {
    const bool ok = !r.hf || r.ours.rel <= 2 * r.fl.rel + 1e-7;
    fails += !ok;
    printf("%-86s %10.3e %10.3e %10.3e %10.3e %5s\n", r.name.c_str(), r.ours.rel, r.hf ? r.fl.rel : NAN, r.ours.maxabs,
           r.hf ? r.fl.maxabs : NAN, ok ? "ok" : "FAIL");
    if (!ok) printf("    worst element %zu: ours %.9g ref %.9g floor-run %.9g\n", r.ours.imax, r.vo, r.vr, r.vf);
  }
  printf("\nloss %.6f, grad norm %.6f, forward+backward of %d samples %.1f ms\n", loss, gn, B, ms);
  printf("%s: %d failing points\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
