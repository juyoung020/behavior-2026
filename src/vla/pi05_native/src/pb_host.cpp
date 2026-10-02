#include "pb_host.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace pi05 {
namespace {

bool read_f64(const WeightFile& wf, const std::string& name, std::vector<double>* out, std::string* err) {
  const TensorInfo* t = wf.find(name);
  if (!t || t->dtype != DType::F64) { *err = "missing f64 tensor " + name; return false; }
  out->resize(t->numel());
  return wf.read(*t, out->data(), err);
}

// b1k_proprio.py R1PRO_COMPACT_PROPRIOCEPTION_INDICES / R1PRO_LEGACY_PROPRIOCEPTION_INDICES
struct Idx { int base_qvel, trunk, arm_l, grip_l, arm_r, grip_r; };
constexpr Idx kCompact{0, 53, 3, 24, 28, 49};
constexpr Idx kLegacy{253, 236, 158, 193, 197, 232};

}  // namespace

bool load_pb_spec(const WeightFile& wf, PbSpec* s, std::string* err) {
  s->ah = wf.cfg_int("action_horizon", 30);
  s->ad = wf.cfg_int("action_dim", 32);
  s->stages.clear();
  if (auto v = wf.cfg("pb.stages"))
    for (auto& x : *v) s->stages.push_back(std::stoi(x));
  s->delta_mask.clear();
  if (auto v = wf.cfg("pb.delta_mask"))
    for (auto& x : *v) s->delta_mask.push_back(std::stoi(x));
  if ((int)s->delta_mask.size() != s->nact) { *err = "pb.delta_mask"; return false; }
  return read_f64(wf, "norm.state.mean", &s->s_mean, err) && read_f64(wf, "norm.state.std", &s->s_std, err) &&
         read_f64(wf, "norm.actions.pt_mean", &s->a_pt_mean, err) && read_f64(wf, "norm.actions.pt_std", &s->a_pt_std, err) &&
         read_f64(wf, "pb.corr_L", &s->corr_L, err);
}

void pb_extract_state(const float* p, int n, float* st) {
  if (n == 23) { memcpy(st, p, 23 * 4); return; }
  const bool legacy = n >= 256;
  const Idx ix = legacy ? kLegacy : kCompact;
  float bq[3] = {p[ix.base_qvel], p[ix.base_qvel + 1], p[ix.base_qvel + 2]};
  if (legacy) {  // world-frame holonomic base velocity -> robot frame (holonomic_base_qvel_to_robot_frame)
    const float yaw = p[246];  // base_qpos 244:247 = [x, y, rz]
    const float c = std::cos(yaw), s = std::sin(yaw);
    const float vx = bq[0], vy = bq[1];
    bq[0] = c * vx + s * vy;
    bq[1] = -s * vx + c * vy;
  }
  int k = 0;
  for (int i = 0; i < 3; ++i) st[k++] = bq[i];
  for (int i = 0; i < 4; ++i) st[k++] = p[ix.trunk + i];
  for (int i = 0; i < 7; ++i) st[k++] = p[ix.arm_l + i];
  st[k++] = 2.0f * ((p[ix.grip_l] + p[ix.grip_l + 1]) / 0.1f) - 1.0f;
  for (int i = 0; i < 7; ++i) st[k++] = p[ix.arm_r + i];
  st[k++] = 2.0f * ((p[ix.grip_r] + p[ix.grip_r + 1]) / 0.1f) - 1.0f;
}

void pb_state_tokens(const PbSpec& s, const float* st, float* norm32, int* tok) {
  for (int i = 0; i < 32; ++i) norm32[i] = 0.f;
  for (int i = 0; i < s.nact; ++i) norm32[i] = (float)(((double)st[i] - s.s_mean[i]) / (s.s_std[i] + 1e-6));
  for (int i = 0; i < 32; ++i) {
    // jnp.digitize(x, linspace(-1, 1, 257)[:-1]) - 1, clipped to [0, 255]  (pi_behavior.py:594-595), float32
    const float x = norm32[i];
    int k;
    if (std::isnan(x)) k = 255;
    else if (x < -1.0f) k = -1;
    else {
      k = (int)std::floor(((double)x + 1.0) * 128.0);
      k = std::min(std::max(k, 0), 255);
      while (k < 255 && x >= -1.0f + (float)(k + 1) / 128.0f) ++k;
      while (k > 0 && x < -1.0f + (float)k / 128.0f) --k;
    }
    tok[i] = std::min(std::max(k, 0), 255);
  }
}

void pb_initial_actions(const PbSpec& s, const double* kept, int keep, const float* st, float* out) {
  for (int t = 0; t < keep; ++t)
    for (int d = 0; d < s.ad; ++d) {
      if (d >= s.nact) { out[t * s.ad + d] = 0.f; continue; }
      double a = kept[t * s.nact + d] - (s.delta_mask[d] ? (double)st[d] : 0.0);  // DeltaActions
      a = (a - s.a_pt_mean[t * s.ad + d]) / (s.a_pt_std[t * s.ad + d] + 1e-6);  // per-timestep z-score
      out[t * s.ad + d] = (float)a;
    }
}

void pb_postprocess(const PbSpec& s, const float* raw, const float* norm32, double* out) {
  double su[32];
  for (int d = 0; d < 32; ++d) su[d] = (double)norm32[d] * (s.s_std[d] + 1e-6) + s.s_mean[d];
  for (int t = 0; t < s.ah; ++t)
    for (int d = 0; d < s.nact; ++d) {
      double a = (double)raw[t * s.ad + d] * (s.a_pt_std[t * s.ad + d] + 1e-6) + s.a_pt_mean[t * s.ad + d];
      if (s.delta_mask[d]) a += su[d];  // AbsoluteActions
      out[t * s.nact + d] = a;
    }
}

void pb_correlate(const PbSpec& s, const float* z, float* noise) {
  const int n = s.ah * s.ad;
  for (int i = 0; i < n; ++i) {
    double acc = 0;
    const double* Li = &s.corr_L[(size_t)i * n];
    for (int j = 0; j <= i; ++j) acc += (double)z[j] * Li[j];  // lower triangular
    noise[i] = (float)acc;
  }
}

// ---- correction_rules.py ---------------------------------------------------------------------------------------------
namespace {
constexpr float OPEN_T = 0.90f, CLOSED_T = -0.98f;
constexpr int LG = 14, RG = 22;
const std::set<int> kOpenLeft = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 48};
const std::set<int> kOpenRight = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 47, 48, 49};
struct MinStage { int left, right; };  // -1 = no rule
MinStage min_stage(int task) {
  switch (task) {
    case 0: return {2, 2};
    case 30: return {-1, 6};
    case 31: return {-1, 8};
    case 32: return {-1, 5};
    case 33: return {-1, 11};
    case 40: return {-1, 4};
    case 41: return {14, 14};
    case 45: return {-1, 10};
    case 46: return {8, 8};
    case 49: return {14, -1};
    default: return {-2, -2};  // not in MIN_STAGE_FOR_CLOSURE
  }
}
void tile_state(const float* st, std::vector<double>& a, int ah) {
  for (int t = 0; t < ah; ++t)
    for (int d = 0; d < 23; ++d) a[t * 23 + d] = (double)st[d];
}
}  // namespace

int pb_correction_rules(int task, int stage, const float* st, std::vector<double>& actions, int ah) {
  // task-specific: radio stage 4 -> 2 and gripper recovery (task0_stage4_reset_to_stage2)
  if (task == 0 && stage >= 2) {
    const int corrected = stage == 4 ? 2 : stage;
    const float l = st[LG], r = st[RG];
    const bool lc = l < CLOSED_T, rc = r < CLOSED_T, lo = l > OPEN_T, ro = r > OPEN_T;
    const bool lmid = !(lo || lc), rmid = !(ro || rc);
    std::vector<double> c(actions.size());
    tile_state(st, c, ah);
    bool change = false;
    if (lc && !rmid) { for (int t = 0; t < ah; ++t) c[t * 23 + LG] = 1.0; change = true; }
    if (rc && !lmid) { for (int t = 0; t < ah; ++t) c[t * 23 + RG] = 1.0; change = true; }
    if (change) { actions = c; return corrected; }
    if (stage != corrected) return corrected;  // actions unchanged
    // rule did not match -> fall through to the general rule
  }
  // general gripper correction
  const float l = st[LG], r = st[RG];
  const bool lc = l < CLOSED_T, rc = r < CLOSED_T;
  const MinStage ms = min_stage(task);
  bool lopen = false, ropen = false;
  if (lc) {
    if (kOpenLeft.count(task)) lopen = true;
    else if (ms.left != -2 && ms.left >= 0 && stage < ms.left) lopen = true;
  }
  if (rc) {
    if (task == 38 || task == 39) {
    } else if (kOpenRight.count(task)) ropen = true;
    else if (ms.right != -2 && ms.right >= 0 && stage < ms.right) ropen = true;
  }
  if (lopen || ropen) {
    std::vector<double> c(actions.size());
    tile_state(st, c, ah);
    if (lopen) for (int t = 0; t < ah; ++t) c[t * 23 + LG] = 1.0;
    if (ropen) for (int t = 0; t < ah; ++t) c[t * 23 + RG] = 1.0;
    actions = c;
  }
  return stage;
}

bool pb_gripper_variation(const std::vector<double>& a, int n) {
  double lmin = 1e300, lmax = -1e300, rmin = 1e300, rmax = -1e300;
  for (int t = 0; t < n; ++t) {
    lmin = std::min(lmin, a[t * 23 + LG]); lmax = std::max(lmax, a[t * 23 + LG]);
    rmin = std::min(rmin, a[t * 23 + RG]); rmax = std::max(rmax, a[t * 23 + RG]);
  }
  return (lmax - lmin) > 0.2 || (rmax - rmin) > 0.2;
}

std::vector<double> pb_cubic_resample(const std::vector<double>& a, int n, int target) {
  // not-a-knot cubic spline through (i, a[i]) (scipy interp1d kind='cubic' = make_interp_spline(k=3)),
  // second-derivative form, knots at integers; evaluated at numpy linspace(0, n-1, target)
  std::vector<double> out((size_t)target * 23);
  std::vector<double> A((size_t)n * n), b(n), M(n);
  std::vector<double> tt(target);
  const double step = (double)(n - 1) / (target - 1);
  for (int k = 0; k < target; ++k) tt[k] = k * step;
  tt[target - 1] = n - 1;
  for (int dim = 0; dim < 23; ++dim) {
    std::fill(A.begin(), A.end(), 0.0);
    auto y = [&](int i) { return a[(size_t)i * 23 + dim]; };
    A[0] = 1; A[1] = -2; A[2] = 1; b[0] = 0;  // M0 - 2 M1 + M2 = 0 (third derivative continuous at x1)
    for (int i = 1; i < n - 1; ++i) {
      A[(size_t)i * n + i - 1] = 1; A[(size_t)i * n + i] = 4; A[(size_t)i * n + i + 1] = 1;
      b[i] = 6 * (y(i - 1) - 2 * y(i) + y(i + 1));
    }
    A[(size_t)(n - 1) * n + n - 3] = 1; A[(size_t)(n - 1) * n + n - 2] = -2; A[(size_t)(n - 1) * n + n - 1] = 1;
    b[n - 1] = 0;
    // Gaussian elimination with partial pivoting (n <= 30)
    std::vector<double> Aw = A, bw = b;
    for (int c = 0; c < n; ++c) {
      int piv = c;
      for (int r = c + 1; r < n; ++r) if (std::fabs(Aw[(size_t)r * n + c]) > std::fabs(Aw[(size_t)piv * n + c])) piv = r;
      if (piv != c) { for (int k = 0; k < n; ++k) std::swap(Aw[(size_t)c * n + k], Aw[(size_t)piv * n + k]); std::swap(bw[c], bw[piv]); }
      for (int r = c + 1; r < n; ++r) {
        const double f = Aw[(size_t)r * n + c] / Aw[(size_t)c * n + c];
        if (f == 0) continue;
        for (int k = c; k < n; ++k) Aw[(size_t)r * n + k] -= f * Aw[(size_t)c * n + k];
        bw[r] -= f * bw[c];
      }
    }
    for (int r = n - 1; r >= 0; --r) {
      double s = bw[r];
      for (int k = r + 1; k < n; ++k) s -= Aw[(size_t)r * n + k] * M[k];
      M[r] = s / Aw[(size_t)r * n + r];
    }
    for (int k = 0; k < target; ++k) {
      const double t = tt[k];
      int i = std::min((int)std::floor(t), n - 2);
      const double u = t - i, w = (i + 1) - t;
      out[(size_t)k * 23 + dim] = M[i] * w * w * w / 6 + M[i + 1] * u * u * u / 6 + (y(i) - M[i] / 6) * w +
                                  (y(i + 1) - M[i + 1] / 6) * u;
    }
  }
  return out;
}

void pb_vote(PbSlot& s, int max_stage, int predicted, const PbWrapperCfg& c) {
  predicted = std::min(predicted, max_stage);
  s.history.push_back(predicted);
  while ((int)s.history.size() > c.history_len) s.history.pop_front();
  if ((int)s.history.size() != c.history_len) return;
  const int next = s.stage + 1;
  if (next > max_stage) return;
  int for_next = 0, skip = 0, back = 0;
  for (int p : s.history) {
    for_next += p == next;
    skip += p == next + 1;
    back += p == s.stage - 1;
  }
  if (for_next >= c.votes_to_promote) { s.stage = next; s.history.clear(); }
  else if (skip == c.history_len) { s.stage = next; s.history.clear(); }
  else if (back == c.history_len && s.stage > 0) { s.stage -= 1; s.history.clear(); }
}

}  // namespace pi05
