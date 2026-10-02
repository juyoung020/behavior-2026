// scenemap C ABI(include/scenemap.h) 구현 — slam2d 까지. 물체 지도(objmap)는 아직 없다(물체 질의는 0 개).
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
#include <vector>

#include "scenemap.h"
#include "scenemap/fk.hpp"
#include "scenemap/slam2d.hpp"

using namespace scenemap;

namespace {
struct Prop {
  double stamp;
  float q[kProprioDim];
};
}  // namespace

struct sm_ctx {
  std::mutex mu;
  SlamParams params;
  Slam2D slam;
  std::vector<std::string> labels;
  std::vector<uint32_t> handled;
  std::deque<Prop> pending;     // 아직 적분하지 않은 proprio(영상 stamp 를 기다림)
  Prop last_used{};             // 가장 최근에 적분한 proprio
  bool have_used = false;
  sm_status st{};
  explicit sm_ctx(const SlamParams& p) : params(p), slam(p) {}
};

struct sm_snapshot_t {
  std::atomic<int> refs{1};
  sm_pose2 pose{};
  sm_status st{};
  double res = 0.05, ox = 0, oy = 0;
  int w = 0, h = 0;
  std::vector<int8_t> cells;
  std::vector<sm_object> objs;
  // reachable 용 부풀린 장애물(처음 부를 때 만듦)
  mutable std::once_flag inflate_once;
  mutable std::vector<uint8_t> blocked;
};

namespace {

// 적분 한 표본(데이터: proprio i 의 base_qvel 이 i-1 → i 구간 속도). 제자리 잡음은 Slam2D 와 같은 규칙.
void integrate(sm_ctx* c, const Prop& p) {
  if (c->have_used) {
    const double dt = p.stamp - c->last_used.stamp;
    if (dt > 0 && dt < 1.0) c->slam.pushVelocity(p.q[0], p.q[1], p.q[2], dt);
  }
  c->last_used = p;
  c->have_used = true;
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
  c->handled.clear();
  c->pending.clear();
  c->have_used = false;
  c->st = sm_status{};
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

int sm_push_image(sm_ctx* c, const sm_image* im, const sm_detections* /*dets: objmap 이 붙으면 씀*/) {
  if (!c || !im || im->w <= 0 || im->h <= 0) return -1;
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
  return 0;
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
  }
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

int sm_snap_find(const sm_snapshot_t* s, const char*, uint32_t*, float*, int) { return s ? 0 : -1; }
int sm_snap_near(const sm_snapshot_t* s, const double*, double, uint32_t*, int) { return s ? 0 : -1; }

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

}  // extern "C"
