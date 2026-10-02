// scenemap C ABI(include/scenemap.h) 구현 — slam2d + objmap(검출 → 물체 지도) + 저장(sm_save_dsg, dsg_save.cpp).
//
// 입력 스레드 하나가 push 를 부르고, 계획기 쪽이 아무 때나 스냅숏을 만든다. 상태는 뮤텍스 하나 아래.
// 스냅숏은 그 순간의 자세·상태·격자(i8)를 통째로 복사한 것(참조 카운트) — 격자 600×600 에서 약 1 ms.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/dsg_save.hpp"
#include "scenemap/fk.hpp"
#include "scenemap/objmap.hpp"
#include "scenemap/slam2d.hpp"

using namespace scenemap;

namespace {
struct Prop {
  double stamp;
  float q[kProprioDim];
};
struct ViewSlot {
  BestViewPtr v;
  double q = 0;                 // 비교에 쓰는 품질(옮겨짐·놓기 때 0 으로)
};
constexpr float kCropMargin = 0.10f;
constexpr int kCropMaxSide = 256;
}  // namespace

struct sm_ctx {
  std::mutex mu;
  SlamParams params;
  Slam2D slam;
  ObjParams oparams;
  ObjectMap om;
  std::vector<std::string> labels;
  std::vector<uint32_t> handled;
  std::deque<Prop> pending;     // 아직 적분하지 않은 proprio(영상 stamp 를 기다림)
  Prop last_used{};             // 가장 최근에 적분한 proprio
  bool have_used = false;
  sm_status st{};
  // best view
  std::unordered_map<uint32_t, ViewSlot> views;
  std::vector<uint32_t> last_assoc;   // 마지막 영상의 검출 → 물체 id
  size_t ev_seen = 0;                 // 처리한 objmap 사건 수(옮겨짐·놓기 → 품질 0)
  uint32_t view_ver = 0;
  uint64_t epoch = 0;                 // sm_reset 마다 +1(잠금 밖 자르기 중 reset 이면 버림)
  explicit sm_ctx(const SlamParams& p) : params(p), slam(p), om(oparams) {}
};

struct sm_snapshot_t {
  std::atomic<int> refs{1};
  sm_pose2 pose{};
  sm_status st{};
  double res = 0.05, ox = 0, oy = 0;
  int w = 0, h = 0;
  std::vector<int8_t> cells;
  std::vector<sm_object> objs;
  std::vector<std::string> names;   // objs[i].name 이 가리키는 문자열(스냅숏 수명 동안)
  std::vector<BestViewPtr> views;   // objs[i] 의 best view(없으면 null)
  // reachable 용 부풀린 장애물(처음 부를 때 만듦)
  mutable std::once_flag inflate_once;
  mutable std::vector<uint8_t> blocked;
};

namespace {

// proprio 의 팔 끝(베이스 기준, 17:20 · 42:45)과 손가락 합(24+25 · 49+50) — docs/scenemap_설계.md 6.1
void handsOf(const float* q, float eef[2][3], float grip[2]) {
  for (int k = 0; k < 3; ++k) { eef[0][k] = q[17 + k]; eef[1][k] = q[42 + k]; }
  grip[0] = q[24] + q[25];
  grip[1] = q[49] + q[50];
}

// 베이스 기준 점 → map (slam 자세)
void toMap(const Pose2& P, const float b[2][3], double m[2][3]) {
  const double c = std::cos(P.th), s = std::sin(P.th);
  for (int k = 0; k < 2; ++k) {
    m[k][0] = P.x + c * b[k][0] - s * b[k][1];
    m[k][1] = P.y + s * b[k][0] + c * b[k][1];
    m[k][2] = b[k][2];
  }
}

// 적분 한 표본(데이터: proprio i 의 base_qvel 이 i-1 → i 구간 속도). 제자리 잡음은 Slam2D 와 같은 규칙.
void integrate(sm_ctx* c, const Prop& p) {
  if (c->have_used) {
    const double dt = p.stamp - c->last_used.stamp;
    if (dt > 0 && dt < 1.0) c->slam.pushVelocity(p.q[0], p.q[1], p.q[2], dt);
  }
  c->last_used = p;
  c->have_used = true;
  // 영상이 없는 스텝에도 든 물체가 손을 따라가게
  float eef[2][3], grip[2];
  handsOf(p.q, eef, grip);
  double eefm[2][3];
  const Pose2 P = c->slam.pose();
  toMap(P, eef, eefm);
  c->om.updateHands(p.stamp, eefm, grip, P.th);
}

Pose2 preview(const sm_ctx* c, double* stamp) {
  Pose2 q = c->slam.pose();
  double t = c->have_used ? c->last_used.stamp : 0;
  for (const Prop& p : c->pending) {
    const double dt = p.stamp - t;
    t = p.stamp;
    if (!(dt > 0 && dt < 1.0)) continue;
    double vx = p.q[0], vy = p.q[1], wz = p.q[2];
    if (c->params.deadband && std::hypot(vx, vy) < c->params.still_v && std::fabs(wz) < c->params.still_w) continue;
    const double cs = std::cos(q.th), sn = std::sin(q.th);
    q.x += (cs * vx - sn * vy) * dt;
    q.y += (sn * vx + cs * vy) * dt;
    q.th += wz * dt;
  }
  *stamp = t;
  return q;
}

}  // namespace

extern "C" {

sm_ctx* sm_create(const char* /*config_json: 아직 기본값만*/) { return new sm_ctx(SlamParams{}); }

void sm_destroy(sm_ctx* c) { delete c; }

int sm_set_labels(sm_ctx* c, const char* const* names, int n) {
  if (!c || n < 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->labels.clear();
  for (int i = 0; i < n; ++i) c->labels.emplace_back(names && names[i] ? names[i] : "");
  return 0;
}

int sm_reset(sm_ctx* c) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->slam = Slam2D(c->params);
  c->om = ObjectMap(c->oparams);
  c->handled.clear();
  c->pending.clear();
  c->have_used = false;
  c->st = sm_status{};
  c->views.clear();
  c->last_assoc.clear();
  c->ev_seen = 0;
  c->epoch++;

  return 0;
}

int sm_push_proprio(sm_ctx* c, const sm_proprio* p) {
  if (!c || !p || !p->proprio || p->n_proprio < kProprioDim) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  Prop e;
  e.stamp = p->stamp;
  std::memcpy(e.q, p->proprio, sizeof(e.q));
  c->pending.push_back(e);
  // 영상이 오지 않아도 쌓이지 않게: 최신보다 0.5 s 넘게 오래된 것은 적분해 둔다(영상 stamp 는 최신 − 1 스텝)
  while (c->pending.size() > 1 && c->pending.front().stamp < p->stamp - 0.5) {
    integrate(c, c->pending.front());
    c->pending.pop_front();
  }
  c->st.last_proprio_stamp = p->stamp;
  c->st.n_proprio++;
  return 0;
}

int sm_push_image(sm_ctx* c, const sm_image* im, const sm_detections* dets) { return sm_push_image_ex(c, im, dets, nullptr, nullptr); }

namespace {
// best view 를 바꿀 검출 하나(잠금 안에서 정하고, 잠금 밖에서 자름)
struct ViewCand {
  uint32_t id;
  double q;
  sm_crop_req req;
  std::shared_ptr<BestView> v;
};
}  // namespace

int sm_push_image_ex(sm_ctx* c, const sm_image* im, const sm_detections* dets, sm_crop_fn crop, void* user) {
  if (!c || !im || im->w <= 0 || im->h <= 0) return -1;
  std::vector<ViewCand> cand;
  uint64_t epoch = 0;
  {
  std::lock_guard<std::mutex> g(c->mu);
  c->st.last_image_stamp = im->stamp;
  c->st.n_images++;
  if (im->cam != 0 || !im->depth_m) return 0;   // slam2d 는 머리 깊이만
  // 영상 stamp 까지 적분(같은 stamp 의 proprio 가 이 영상의 짝)
  const double eps = 1e-6;
  while (!c->pending.empty() && c->pending.front().stamp <= im->stamp + eps) {
    integrate(c, c->pending.front());
    c->pending.pop_front();
  }
  if (!c->have_used) return 0;                  // 짝지을 proprio 가 아직 없음
  BodyFk fk;
  computeBodyFk(c->last_used.q, &fk);
  const float eef[2][3] = {{c->last_used.q[17], c->last_used.q[18], c->last_used.q[19]},
                           {c->last_used.q[42], c->last_used.q[43], c->last_used.q[44]}};
  const BodyState body = bodyFromFk(fk, eef);
  DepthView dv;
  dv.w = im->w;
  dv.h = im->h;
  dv.m = im->depth_m;
  dv.step = std::max(1, int(std::lround(im->w / 160.0)));   // 채점과 같은 표본 밀도(가로 160 점 안팎)
  dv.fx = float(im->fx); dv.fy = float(im->fy); dv.cx = float(im->cx); dv.cy = float(im->cy);
  std::memcpy(dv.T_bc, fk.T_head, sizeof(dv.T_bc));
  c->slam.keyframe(dv, body);
  if (!dets) { c->last_assoc.clear(); return 0; }   // 검출 없음: 지도(slam2d)만
  // 물체 지도: map ← 카메라 = slam 자세 ∘ 순기구학 머리 카메라(objmap_eval 과 같은 계산)
  const Pose2 P = c->slam.pose();
  const double cs = std::cos(P.th), sn = std::sin(P.th);
  ObjFrame F;
  F.stamp = im->stamp;
  F.w = im->w;
  F.h = im->h;
  F.depth_m = im->depth_m;
  F.fx = float(im->fx); F.fy = float(im->fy); F.cx = float(im->cx); F.cy = float(im->cy);
  const float* B = fk.T_head;
  for (int r = 0; r < 3; ++r) {
    const double R0 = r == 0 ? cs : (r == 1 ? sn : 0), R1 = r == 0 ? -sn : (r == 1 ? cs : 0), R2 = r == 2 ? 1 : 0;
    for (int k = 0; k < 4; ++k) F.T_mc[r * 4 + k] = R0 * B[k] + R1 * B[4 + k] + R2 * B[8 + k];
  }
  F.T_mc[3] += P.x;
  F.T_mc[7] += P.y;
  F.dets = dets;
  float eefb[2][3], grip[2];
  handsOf(c->last_used.q, eefb, grip);
  toMap(P, eefb, F.eef);
  F.grip[0] = grip[0];
  F.grip[1] = grip[1];
  F.base_yaw = P.th;
  c->om.update(F);
  // 검출 → 물체 id
  const std::vector<DetAssoc>& as = c->om.lastAssoc();
  c->last_assoc.resize(as.size());
  for (size_t k = 0; k < as.size(); ++k) c->last_assoc[k] = as[k].obj_id;
  // 옮겨짐·놓기: 지금 모습은 옛 자리 — 품질을 내려 다음 관측이 바꾸게
  const auto& ev = c->om.events();
  for (; c->ev_seen < ev.size(); ++c->ev_seen)
    if (ev[c->ev_seen].kind == 2 || ev[c->ev_seen].kind == 5) {
      auto it = c->views.find(ev[c->ev_seen].id);
      if (it != c->views.end()) it->second.q = 0;
    }
  // 없어진 물체(버린 후보)의 모습 버리기
  if (!c->views.empty()) {
    std::unordered_set<uint32_t> live;
    for (const MapObject& o : c->om.objects()) live.insert(o.id);
    for (auto it = c->views.begin(); it != c->views.end();) it = live.count(it->first) ? std::next(it) : c->views.erase(it);
  }
  // 새 모습 후보: 품질 = 유효 마스크 넓이 × 점수, 지금 것 이상(같으면 최근)
  const bool host_rgb = !crop && im->rgba && dets->img_w == im->w && dets->img_h == im->h;
  if (!crop && !host_rgb) return 0;
  for (size_t k = 0; k < as.size(); ++k) {
    if (!as[k].obj_id) continue;
    const float sc = dets->score ? dets->score[k] : 1.f;
    const double q = double(as[k].area_px) * sc;
    auto it = c->views.find(as[k].obj_id);
    if (it != c->views.end() && q < it->second.q) continue;
    ViewCand vc{};
    vc.id = as[k].obj_id;
    vc.q = q;
    auto v = std::make_shared<BestView>();
    const float* b = dets->box + 4 * k;
    if (!cropGeometry(b, dets->img_w, dets->img_h, kCropMargin, kCropMaxSide, v->box, &v->w, &v->h)) continue;
    v->id = vc.id;
    v->q = q;
    v->stamp = im->stamp;
    for (int i = 0; i < 4; ++i) v->det_box[i] = int32_t(std::lround(b[i]));
    v->mask_area = as[k].area_px;
    v->depth_m = as[k].depth_med;
    v->score = sc;
    std::memcpy(v->cam_T, F.T_mc, sizeof(v->cam_T));
    v->rgb.resize(size_t(v->w) * v->h * 3);
    v->depth.resize(size_t(v->w) * v->h);
    vc.req = sm_crop_req{v->box[0], v->box[1], v->box[2], v->box[3], v->w, v->h, v->rgb.data()};
    vc.v = std::move(v);
    cand.push_back(std::move(vc));
  }
  epoch = c->epoch;
  }  // 잠금 끝: 자르기(장치 → 호스트 복사)는 잠금 밖에서
  if (cand.empty()) return 0;
  std::vector<sm_crop_req> reqs(cand.size());
  for (size_t i = 0; i < cand.size(); ++i) reqs[i] = cand[i].req;
  if (crop) {
    if (crop(user, reqs.data(), int32_t(reqs.size())) != 0) return 0;   // 자르기 실패: 이번 모습은 버림
  } else {
    for (const sm_crop_req& r : reqs) cropRgbHost(im->rgba, int64_t(im->w) * 4, 4, r);
  }
  for (ViewCand& vc : cand)
    cropDepthMm(im->depth_m, im->w, im->h, dets->img_w, dets->img_h, vc.v->box, vc.v->w, vc.v->h, vc.v->depth.data());
  std::lock_guard<std::mutex> g(c->mu);
  if (c->epoch != epoch) return 0;
  for (ViewCand& vc : cand) {
    ViewSlot& sl = c->views[vc.id];
    if (sl.v && vc.q < sl.q) continue;
    vc.v->version = ++c->view_ver;
    sl.v = std::move(vc.v);
    sl.q = vc.q;
  }
  return 0;
}

int sm_last_assoc(sm_ctx* c, uint32_t* ids, int cap) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  const int n = int(c->last_assoc.size());
  for (int k = 0; k < std::min(n, cap) && ids; ++k) ids[k] = c->last_assoc[k];
  return n;
}

int sm_mark_handled(sm_ctx* c, uint32_t id) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (std::find(c->handled.begin(), c->handled.end(), id) == c->handled.end()) c->handled.push_back(id);
  return 0;
}

int sm_snapshot(sm_ctx* c, sm_snapshot_t** out) {
  if (!c || !out) return -1;
  auto* s = new sm_snapshot_t();
  {
    std::lock_guard<std::mutex> g(c->mu);
    double t;
    const Pose2 q = preview(c, &t);
    s->pose = sm_pose2{t, q.x, q.y, q.th};
    s->st = c->st;
    const OccGrid& gr = c->slam.grid();
    s->res = gr.res();
    s->ox = gr.x0() * double(gr.res());
    s->oy = gr.y0() * double(gr.res());
    s->w = gr.width();
    s->h = gr.height();
    s->cells = gr.export8();
    for (const MapObject& o : c->om.objects()) {
      if (!o.confirmed) continue;
      s->names.push_back(o.cls >= 0 && size_t(o.cls) < c->labels.size() ? c->labels[o.cls] : std::string("?"));
      sm_object e{};
      e.id = o.id;
      e.score = o.score;
      for (int k = 0; k < 3; ++k) { e.pos[k] = o.pos[k]; e.extent[k] = o.ext[k]; e.first_pos[k] = o.first_pos[k]; }
      e.n_obs = o.n_obs;
      e.last_seen = o.last_seen;
      e.state = o.held_by >= 0 ? SM_HELD : o.state;
      e.handled = std::find(c->handled.begin(), c->handled.end(), o.id) != c->handled.end();
      e.structural = std::max({o.ext[0], o.ext[1], o.ext[2]}) > c->oparams.big;
      s->objs.push_back(e);
      auto it = c->views.find(o.id);
      s->views.push_back(it != c->views.end() ? it->second.v : nullptr);
    }
  }
  for (size_t i = 0; i < s->objs.size(); ++i) s->objs[i].name = s->names[i].c_str();
  s->st.n_objects = int32_t(s->objs.size());
  *out = s;
  return 0;
}

void sm_snapshot_release(sm_snapshot_t* s) {
  if (s && s->refs.fetch_sub(1) == 1) delete s;
}

sm_pose2 sm_snap_pose(const sm_snapshot_t* s) { return s ? s->pose : sm_pose2{}; }
sm_status sm_snap_status(const sm_snapshot_t* s) { return s ? s->st : sm_status{}; }

int sm_snap_objects(const sm_snapshot_t* s, const sm_object** out) {
  if (!s || !out) return -1;
  *out = s->objs.data();
  return int(s->objs.size());
}

// 이름 부분 일치(대소문자·'_'↔' ' 무시), 점수 = 일치 길이 비율 × 관측 신뢰도. 점수 순.
int sm_snap_find(const sm_snapshot_t* s, const char* name, uint32_t* ids, float* scores, int cap) {
  if (!s || !name) return -1;
  auto norm = [](std::string t) {
    for (char& ch : t) ch = ch == '_' ? ' ' : char(std::tolower(static_cast<unsigned char>(ch)));
    return t;
  };
  const std::string q = norm(name);
  std::vector<std::pair<float, uint32_t>> hit;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    const std::string n = norm(s->names[i]);
    if (q.empty() || n.find(q) == std::string::npos) continue;
    hit.push_back({float(q.size()) / float(std::max<size_t>(1, n.size())) * std::max(0.05f, s->objs[i].score), s->objs[i].id});
  }
  std::sort(hit.begin(), hit.end(), [](auto& a, auto& b) { return a.first > b.first; });
  const int m = std::min<int>(cap, int(hit.size()));
  for (int i = 0; i < m; ++i) {
    if (ids) ids[i] = hit[i].second;
    if (scores) scores[i] = hit[i].first;
  }
  return m;
}

// p 에서 r 안의 물체(중심 거리), 가까운 순.
int sm_snap_near(const sm_snapshot_t* s, const double p[3], double r, uint32_t* ids, int cap) {
  if (!s || !p) return -1;
  std::vector<std::pair<double, uint32_t>> hit;
  for (const sm_object& o : s->objs) {
    const double d = std::sqrt((o.pos[0] - p[0]) * (o.pos[0] - p[0]) + (o.pos[1] - p[1]) * (o.pos[1] - p[1]) + (o.pos[2] - p[2]) * (o.pos[2] - p[2]));
    if (d <= r) hit.push_back({d, o.id});
  }
  std::sort(hit.begin(), hit.end());
  const int m = std::min<int>(cap, int(hit.size()));
  for (int i = 0; i < m && ids; ++i) ids[i] = hit[i].second;
  return m;
}

int sm_snap_view(const sm_snapshot_t* s, uint32_t id, sm_view* out) {
  if (!s || !out) return -1;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    if (s->objs[i].id != id) continue;
    const BestView* v = s->views[i].get();
    if (!v) return 0;
    sm_view& o = *out;
    o = sm_view{};
    o.id = v->id;
    o.version = v->version;
    o.stamp = v->stamp;
    std::memcpy(o.box_px, v->box, sizeof(o.box_px));
    std::memcpy(o.det_box_px, v->det_box, sizeof(o.det_box_px));
    o.mask_area = v->mask_area;
    o.depth_m = v->depth_m;
    o.score = v->score;
    std::memcpy(o.cam_T, v->cam_T, sizeof(o.cam_T));
    o.w = v->w;
    o.h = v->h;
    o.rgb = v->rgb.empty() ? nullptr : v->rgb.data();
    o.depth_mm = v->depth.empty() ? nullptr : v->depth.data();
    return 1;
  }
  return 0;
}

int sm_snap_map(const sm_snapshot_t* s, sm_grid* out) {
  if (!s || !out) return -1;
  out->resolution = s->res;
  out->origin[0] = s->ox;
  out->origin[1] = s->oy;
  out->width = s->w;
  out->height = s->h;
  out->cells = s->cells.data();
  return 0;
}

// 격자 위 8방향 A*. 점유(≥ 65 %) 칸을 로봇 반경 0.30 m 만큼 부풀려 막는다. 모르는 칸은 지나갈 수 있되 1.5 배 비용.
// 목표: `to` 에서 0.6 m 안의 막히지 않은 칸(물체는 가구 위에 있으니 그 앞까지). 출발 칸이 부풀림 안이면 0.4 m 안은 풀어 준다.
// 두 점 중 하나라도 지도 밖이면 직선 거리(모름).
double sm_snap_reachable(const sm_snapshot_t* s, const double from[2], const double to[2]) {
  if (!s || !from || !to) return -1;
  const double straight = std::hypot(to[0] - from[0], to[1] - from[1]);
  const int W = s->w, H = s->h;
  auto cell = [&](const double* p, int* x, int* y) {
    *x = int(std::floor((p[0] - s->ox) / s->res));
    *y = int(std::floor((p[1] - s->oy) / s->res));
    return *x >= 0 && *y >= 0 && *x < W && *y < H;
  };
  int sx, sy, gx, gy;
  if (!W || !cell(from, &sx, &sy) || !cell(to, &gx, &gy)) return straight;
  std::call_once(s->inflate_once, [&] {
    s->blocked.assign(size_t(W) * H, 0);
    const int r = int(std::ceil(0.30 / s->res));
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (s->cells[size_t(y) * W + x] < 65) continue;
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            const int X = x + dx, Y = y + dy;
            if (X < 0 || Y < 0 || X >= W || Y >= H || dx * dx + dy * dy > r * r) continue;
            s->blocked[size_t(Y) * W + X] = 1;
          }
      }
  });
  const int rs = int(std::ceil(0.40 / s->res)), rg = int(std::ceil(0.60 / s->res));
  auto isBlocked = [&](int x, int y) {
    if ((x - sx) * (x - sx) + (y - sy) * (y - sy) <= rs * rs) return false;
    return s->blocked[size_t(y) * W + x] != 0;
  };
  auto isGoal = [&](int x, int y) { return (x - gx) * (x - gx) + (y - gy) * (y - gy) <= rg * rg; };
  std::vector<float> dist(size_t(W) * H, INFINITY);
  using QE = std::pair<float, int>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
  auto hcost = [&](int x, int y) { return float(std::hypot(x - gx, y - gy)); };
  dist[size_t(sy) * W + sx] = 0;
  pq.push({hcost(sx, sy), sy * W + sx});
  const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (!pq.empty()) {
    const auto [f, id] = pq.top();
    pq.pop();
    const int x = id % W, y = id / W;
    const float d = dist[id];
    if (f > d + hcost(x, y) + 1e-3f) continue;
    if (isGoal(x, y)) {
      const double rest = std::max(0.0, std::hypot(x - gx, y - gy) * s->res);   // 목표 둘레에서 멈춘 만큼(가구 위 물체까지)
      return d * s->res + rest;
    }
    for (int k = 0; k < 8; ++k) {
      const int X = x + DX[k], Y = y + DY[k];
      if (X < 0 || Y < 0 || X >= W || Y >= H || isBlocked(X, Y)) continue;
      const float step = (k < 4 ? 1.f : 1.41421356f) * (s->cells[size_t(Y) * W + X] < 0 ? 1.5f : 1.f);
      const size_t j = size_t(Y) * W + X;
      if (d + step < dist[j]) {
        dist[j] = d + step;
        pq.push({dist[j] + hcost(X, Y), int(j)});
      }
    }
  }
  return -1;
}

int sm_save_dsg(sm_ctx* c, const char* dir) {
  if (!c || !dir) return -1;
  sm_snapshot_t* s = nullptr;
  if (sm_snapshot(c, &s) != 0) return -1;
  SaveInput in;
  in.stamp = s->pose.stamp;
  in.pose[0] = s->pose.x; in.pose[1] = s->pose.y; in.pose[2] = s->pose.yaw;
  in.objs = s->objs.data();
  in.n_objs = int(s->objs.size());
  in.grid_res = s->res; in.grid_ox = s->ox; in.grid_oy = s->oy; in.grid_w = s->w; in.grid_h = s->h;
  in.cells = s->cells.data();
  {
    std::lock_guard<std::mutex> g(c->mu);
    const auto& ev = c->om.events();
    in.events.assign(ev.end() - std::min<size_t>(ev.size(), 50), ev.end());   // 최근 50 개
  }
  const int rc = saveScene(in, dir);
  sm_snapshot_release(s);
  return rc;
}

}  // extern "C"
