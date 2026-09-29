#include "host_io.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace pi05 {
namespace {

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
  return v;
}

bool read_f64(const WeightFile& wf, const std::string& name, std::vector<double>* out, std::string* err) {
  const TensorInfo* t = wf.find(name);
  if (!t || t->dtype != DType::F64) { *err = "missing f64 tensor " + name; return false; }
  out->resize(t->numel());
  return wf.read(*t, out->data(), err);
}

// UTF-8 decode one code point at s[i]; returns length (1 on malformed).
size_t utf8_cp(const std::string& s, size_t i, uint32_t* cp) {
  unsigned char c = s[i];
  size_t n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
  if (i + n > s.size()) n = 1;
  if (n == 1) { *cp = c; return 1; }
  uint32_t v = c & (0x7F >> n);
  for (size_t k = 1; k < n; ++k) v = (v << 6) | (s[i + k] & 0x3F);
  *cp = v;
  return n;
}

bool py_isspace(uint32_t c) {  // str.isspace()
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

}  // namespace

int RobotSpec::max_proprio_index() const {
  int m = -1;
  for (auto& p : proprio)
    for (int i : p.second) m = std::max(m, i);
  return m;
}

bool load_robot_spec(const WeightFile& wf, RobotSpec* rs, std::string* err) {
  auto pv = wf.cfg("robot.proprio");
  if (!pv) { *err = "no robot.proprio"; return false; }
  for (auto& e : *pv) {
    auto c = e.find(':');
    rs->proprio.push_back({e.substr(0, c) == "sum", parse_ints(e.substr(c + 1))});
  }
  for (auto it = wf.cfg_all().lower_bound("robot.delta"); it != wf.cfg_all().upper_bound("robot.delta"); ++it) {
    const std::string& e = it->second[0];
    auto c = e.find(':');
    rs->delta.push_back({parse_ints(e.substr(0, c)), parse_ints(e.substr(c + 1))});
  }
  if (auto g = wf.cfg("robot.grippers"))
    for (auto& s : *g) rs->grippers.push_back(std::stoi(s));
  for (int i = 0; i < 3; ++i) rs->cams.push_back(wf.cfg_str("robot.cam" + std::to_string(i), ""));
  rs->name = wf.cfg_str("robot.name", "robot");
  rs->action_dim = wf.cfg_int("robot.action_dim", 23);
  return read_f64(wf, "norm.state.mean", &rs->s_mean, err) && read_f64(wf, "norm.state.std", &rs->s_std, err) &&
         read_f64(wf, "norm.actions.mean", &rs->a_mean, err) && read_f64(wf, "norm.actions.std", &rs->a_std, err);
}

std::string clean_prompt(const std::string& s) {
  // strip Python whitespace (code points), then "_" -> " " and "\n" -> " "
  size_t b = 0, e = s.size();
  while (b < e) {
    uint32_t cp;
    size_t n = utf8_cp(s, b, &cp);
    if (!py_isspace(cp)) break;
    b += n;
  }
  while (e > b) {  // walk back to the start of the last code point
    size_t k = e - 1;
    while (k > b && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
    uint32_t cp;
    utf8_cp(s, k, &cp);
    if (!py_isspace(cp)) break;
    e = k;
  }
  std::string r = s.substr(b, e - b);
  for (char& c : r)
    if (c == '_' || c == '\n') c = ' ';
  return r;
}

int discretize_state(double x) {
  // bins[i] = -1 + i/128 exactly (np.linspace(-1, 1, 257)[:-1]); digitize(right=False) - 1 = #(bins <= x) - 1
  if (std::isnan(x)) return 255;  // NaN sorts after every bin
  if (x < -1.0) return -1;
  int k = (int)std::floor((x + 1.0) * 128.0);
  if (k > 255) k = 255;
  if (k < 0) k = 0;
  while (k < 255 && x >= -1.0 + (k + 1) / 128.0) ++k;
  while (k > 0 && x < -1.0 + k / 128.0) --k;
  return k;
}

void prepare_input(const RobotSpec& rs, const Tokenizer& tok, const float* proprio, int n_proprio,
                   const std::string& prompt, int max_len, int model_action_dim, PreparedInput* out) {
  // extract_state: grippers are summed in float32 (numpy sum over two f32 values)
  std::vector<float> st;
  for (auto& p : rs.proprio) {
    if (p.first) {
      float s = 0.f;
      for (int i : p.second) s = (i < n_proprio) ? s + proprio[i] : s;
      st.push_back(s);
    } else {
      for (int i : p.second) st.push_back(i < n_proprio ? proprio[i] : 0.f);
    }
  }
  const int n = (int)st.size();
  std::vector<double> norm(n);
  for (int i = 0; i < n; ++i) norm[i] = ((double)st[i] - rs.s_mean[i]) / (rs.s_std[i] + 1e-6);
  std::string state_str;
  for (int i = 0; i < n; ++i) {
    if (i) state_str += ' ';
    state_str += std::to_string(discretize_state(norm[i]));
  }
  out->prompt_text = "Task: " + clean_prompt(prompt) + ", State: " + state_str + ";\nAction: ";
  out->tokens = tok.encode(out->prompt_text, true);
  out->truncated = (int)out->tokens.size() > max_len;
  if (out->truncated) {
    fprintf(stderr, "pi05: token length %zu exceeds %d, truncating\n", out->tokens.size(), max_len);
    out->tokens.resize(max_len);
  }
  out->state_f32.assign(model_action_dim, 0.f);
  for (int i = 0; i < n && i < model_action_dim; ++i) out->state_f32[i] = (float)norm[i];
}

void postprocess_actions(const RobotSpec& rs, const float* raw, int horizon, int model_action_dim,
                         const std::vector<float>& state_f32, double* out) {
  const int n = (int)rs.a_mean.size();
  std::vector<double> state(model_action_dim);
  for (int i = 0; i < model_action_dim; ++i) {
    double sd = i < (int)rs.s_std.size() ? rs.s_std[i] : 1.0, mu = i < (int)rs.s_mean.size() ? rs.s_mean[i] : 0.0;
    state[i] = (double)state_f32[i] * (sd + 1e-6) + mu;
  }
  std::vector<double> a(model_action_dim);
  for (int t = 0; t < horizon; ++t) {
    for (int i = 0; i < model_action_dim; ++i) {
      double sd = i < n ? rs.a_std[i] : 1.0, mu = i < n ? rs.a_mean[i] : 0.0;
      a[i] = (double)raw[t * model_action_dim + i] * (sd + 1e-6) + mu;
    }
    for (auto& d : rs.delta)
      for (size_t k = 0; k < d.first.size(); ++k) a[d.first[k]] += state[d.second[k]];
    for (int i = 0; i < rs.action_dim; ++i) out[t * rs.action_dim + i] = a[i];
  }
}

}  // namespace pi05
