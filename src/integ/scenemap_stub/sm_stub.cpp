// scenemap 가짜 구현 — sm_api.h 에 맞춘 얇은 판. 진짜 libscenemap(slam2d·objmap·YOLOE)이 붙기 전까지 simlink·계획기 연결과
// cargo test 를 돌리려고 둔다. 자세는 base_qvel 적분(계획기 odom.rs 와 같은 식), 물체는 시험 훅(sm_stub_add_object)으로만 생긴다.
// 스레드: 모든 함수가 한 뮤텍스 아래(가짜라 단순하게), 스냅숏은 통째 복사 + 참조 카운트.
#include "sm_api.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Obj {
  sm_object o;
  std::string name;
};
struct State {
  sm_pose2 pose{0, 0, 0, 0};
  double last_qvel[3] = {0, 0, 0};
  bool have_qvel = false;
  sm_status st{};
  std::vector<Obj> objs;
  std::vector<std::string> labels;
};
}  // namespace

struct sm_snapshot_t {
  std::atomic<int> refs{1};
  State s;
  std::vector<sm_object> view;
  std::vector<int8_t> cells;
};

struct sm_ctx {
  std::mutex mu;
  State s;
  double hz = 30.0;
};

static void rebuild_view(sm_snapshot_t* sn) {
  sn->view.clear();
  for (auto& o : sn->s.objs) {
    sm_object v = o.o;
    v.name = o.name.c_str();
    sn->view.push_back(v);
  }
}

static std::string lower(const std::string& a) {
  std::string r = a;
  for (auto& c : r) c = (char)std::tolower((unsigned char)c);
  return r;
}

extern "C" {

sm_ctx* sm_create(const char* /*config_json*/) { return new sm_ctx(); }
void sm_destroy(sm_ctx* c) { delete c; }

int sm_set_labels(sm_ctx* c, const char* const* names, int n) {
  std::lock_guard<std::mutex> g(c->mu);
  c->s.labels.clear();
  for (int i = 0; i < n; ++i) c->s.labels.emplace_back(names[i] ? names[i] : "");
  return 0;
}

int sm_reset(sm_ctx* c) {
  std::lock_guard<std::mutex> g(c->mu);
  auto labels = c->s.labels;
  c->s = State{};
  c->s.labels = labels;
  return 0;
}

int sm_push_proprio(sm_ctx* c, const sm_proprio* p) {
  if (!p || !p->proprio || p->n_proprio < 3) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  State& s = c->s;
  const double dt = 1.0 / c->hz;
  if (s.have_qvel) {  // 프레임 i 는 i-1 의 속도로 한 스텝 전진(odom.rs 와 같음)
    const double cy = std::cos(s.pose.yaw), sy = std::sin(s.pose.yaw);
    s.pose.x += (cy * s.last_qvel[0] - sy * s.last_qvel[1]) * dt;
    s.pose.y += (sy * s.last_qvel[0] + cy * s.last_qvel[1]) * dt;
    s.pose.yaw += s.last_qvel[2] * dt;
  }
  for (int i = 0; i < 3; ++i) s.last_qvel[i] = p->proprio[i];
  s.have_qvel = true;
  s.pose.stamp = p->stamp;
  s.st.last_proprio_stamp = p->stamp;
  s.st.n_proprio++;
  return 0;
}

int sm_push_image(sm_ctx* c, const sm_image* im, const sm_detections* /*dets*/) {
  if (!im || im->w <= 0 || im->h <= 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->s.st.last_image_stamp = im->stamp;
  c->s.st.n_images++;
  return 0;
}

int sm_mark_handled(sm_ctx* c, uint32_t id) {
  std::lock_guard<std::mutex> g(c->mu);
  for (auto& o : c->s.objs)
    if (o.o.id == id) o.o.handled = 1;
  return 0;
}

int sm_snapshot(sm_ctx* c, sm_snapshot_t** out) {
  auto* sn = new sm_snapshot_t();
  {
    std::lock_guard<std::mutex> g(c->mu);
    sn->s = c->s;
  }
  sn->s.st.n_objects = (int32_t)sn->s.objs.size();
  rebuild_view(sn);
  *out = sn;
  return 0;
}

void sm_snapshot_release(sm_snapshot_t* sn) {
  if (sn && sn->refs.fetch_sub(1) == 1) delete sn;
}

sm_pose2 sm_snap_pose(const sm_snapshot_t* sn) { return sn->s.pose; }
sm_status sm_snap_status(const sm_snapshot_t* sn) { return sn->s.st; }

int sm_snap_objects(const sm_snapshot_t* sn, const sm_object** out) {
  *out = sn->view.data();
  return (int)sn->view.size();
}

int sm_snap_find(const sm_snapshot_t* sn, const char* name, uint32_t* ids, float* scores, int cap) {
  const std::string q = lower(name ? name : "");
  int k = 0;
  for (auto& o : sn->s.objs) {
    const std::string n = lower(o.name);
    if (q.empty() || k >= cap) continue;
    if (n.find(q) != std::string::npos || q.find(n) != std::string::npos) {
      ids[k] = o.o.id;
      if (scores) scores[k] = 1.0f;
      ++k;
    }
  }
  return k;
}

int sm_snap_near(const sm_snapshot_t* sn, const double p[3], double r, uint32_t* ids, int cap) {
  std::vector<std::pair<double, uint32_t>> v;
  for (auto& o : sn->s.objs) {
    double d = 0;
    for (int i = 0; i < 3; ++i) d += (o.o.pos[i] - p[i]) * (o.o.pos[i] - p[i]);
    d = std::sqrt(d);
    if (d <= r) v.push_back({d, o.o.id});
  }
  std::sort(v.begin(), v.end());
  int k = 0;
  for (auto& x : v)
    if (k < cap) ids[k++] = x.second;
  return k;
}

int sm_snap_map(const sm_snapshot_t* sn, sm_grid* out) {
  out->resolution = 0.05;
  out->origin[0] = out->origin[1] = 0;
  out->width = out->height = 0;
  out->cells = sn->cells.data();
  return 0;
}

double sm_snap_reachable(const sm_snapshot_t*, const double from[2], const double to[2]) {
  return std::hypot(from[0] - to[0], from[1] - to[1]);
}

// ---- 가짜 전용(ABI 아님): 시험에서 물체 넣기 ----
int sm_stub_add_object(sm_ctx* c, uint32_t id, const char* name, double x, double y, double z) {
  std::lock_guard<std::mutex> g(c->mu);
  Obj o{};
  o.name = name ? name : "";
  o.o.id = id;
  o.o.score = 1.0f;
  o.o.pos[0] = o.o.first_pos[0] = x;
  o.o.pos[1] = o.o.first_pos[1] = y;
  o.o.pos[2] = o.o.first_pos[2] = z;
  o.o.n_obs = 1;
  c->s.objs.push_back(o);
  return 0;
}

int sm_stub_is_stub(void) { return 1; }

}  // extern "C"
