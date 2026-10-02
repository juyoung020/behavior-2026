// sgrt 구현(include/sgrt.h): ovdet + scenemap + 주기 저장을 한 C ABI 로.
#include "sgrt.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ovdet.h"
#include "scenemap.h"

struct sgrt {
  sgrt_config cfg{};
  std::string out_dir;
  OvdHandle* det = nullptr;
  sm_ctx* sm = nullptr;
  int64_t step = 0;
  double last_save = -1e9;
  int32_t n_kf = 0, n_det = 0;
  float det_ms = 0, save_ms = 0;
};

namespace {
void put(char* err, size_t n, const char* msg) {
  if (err && n) std::snprintf(err, n, "%s", msg);
}
double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

extern "C" {

void sgrt_default_config(sgrt_config* c) {
  if (!c) return;
  *c = sgrt_config{};
  c->kf_every = 6;
  c->save_s = 1.0;
  c->conf_th = 0.25f;
}

sgrt* sgrt_create(const sgrt_config* c, char* err, size_t err_len) {
  if (!c || !c->engine || !c->names || !c->out_dir) {
    put(err, err_len, "sgrt_create: engine, names and out_dir are required");
    return nullptr;
  }
  auto* s = new sgrt();
  s->cfg = *c;
  s->out_dir = c->out_dir;
  if (s->cfg.kf_every < 1) s->cfg.kf_every = 1;
  OvdConfig oc;
  ovd_default_config(&oc);
  oc.seg_engine = c->engine;
  oc.names = c->names;
  if (c->conf_th > 0) oc.conf_th = c->conf_th;
  s->det = ovd_create(&oc, err, err_len);
  if (!s->det) {
    delete s;
    return nullptr;
  }
  s->sm = sm_create(nullptr);
  return s;
}

void sgrt_destroy(sgrt* s) {
  if (!s) return;
  if (s->det) ovd_destroy(s->det);
  if (s->sm) sm_destroy(s->sm);
  delete s;
}

int sgrt_begin(sgrt* s, const char* const* prompt, int32_t n, char* err, size_t err_len) {
  if (!s) return -1;
  ovd_set_prompt(s->det, prompt, n, err, err_len);   // 어휘 밖 이름은 err 에 적히고 번호는 유지(검출 안 됨)
  sm_reset(s->sm);
  sm_set_labels(s->sm, prompt, n);                   // 같은 순서 = 검출 cls 가 그대로 이름 번호
  s->step = 0;
  s->last_save = -1e9;
  s->n_kf = s->n_det = 0;
  return 0;
}

int sgrt_want_image(const sgrt* s) { return s && (s->step % s->cfg.kf_every) == 0; }

int sgrt_step(sgrt* s, double stamp, const float* proprio, int32_t n_proprio, const uint8_t* rgb, int32_t rgb_on_device,
              int64_t row_stride, int32_t pix_stride, int32_t w, int32_t h, const float* depth_m, double fx, double fy, double cx,
              double cy) {
  if (!s || !proprio) return -1;
  sm_proprio p{stamp, proprio, n_proprio};
  int rc = sm_push_proprio(s->sm, &p);
  if (rc == 0 && rgb && depth_m && w > 0 && h > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    OvdImage im{};
    im.stamp = stamp;
    im.cam = 0;
    im.data = rgb;
    im.w = w;
    im.h = h;
    im.row_stride = row_stride;
    im.pix_stride = pix_stride;
    im.on_device = rgb_on_device;
    const sm_detections* d = ovd_detect(s->det, &im, nullptr);
    s->det_ms = float(msSince(t0));
    sm_image si{};
    si.stamp = stamp;
    si.cam = 0;
    si.w = w;
    si.h = h;
    si.depth_m = depth_m;
    si.fx = fx; si.fy = fy; si.cx = cx; si.cy = cy;
    rc = sm_push_image(s->sm, &si, d);
    s->n_kf++;
    s->n_det = d ? d->n : 0;
  }
  s->step++;
  if (stamp - s->last_save >= s->cfg.save_s) sgrt_save(s), s->last_save = stamp;
  return rc;
}

int sgrt_save(sgrt* s) {
  if (!s) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = sm_save_dsg(s->sm, s->out_dir.c_str());
  s->save_ms = float(msSince(t0));
  return rc;
}

void sgrt_stats(const sgrt* s, int32_t* n_kf, int32_t* n_det, int32_t* n_obj, float* det_ms, float* save_ms) {
  if (!s) return;
  if (n_kf) *n_kf = s->n_kf;
  if (n_det) *n_det = s->n_det;
  if (n_obj) {
    sm_snapshot_t* snap = nullptr;
    *n_obj = sm_snapshot(s->sm, &snap) == 0 ? sm_snap_status(snap).n_objects : 0;
    sm_snapshot_release(snap);
  }
  if (det_ms) *det_ms = s->det_ms;
  if (save_ms) *save_ms = s->save_ms;
}

}  // extern "C"
