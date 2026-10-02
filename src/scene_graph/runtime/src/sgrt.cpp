// sgrt 구현(include/sgrt.h): ovdet + scenemap + 주기 저장을 한 C ABI 로.
#include "sgrt.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "crop.hpp"
#include "ovdet.h"
#include "scenemap.h"
#include "scenemap/bestview.hpp"

constexpr int kClosedVocabMax = 200;   // 이 이하 어휘 = 닫힌 어휘 엔진(COCO-80), 기본으로 어휘 전부

struct sgrt {
  sgrt_config cfg{};
  std::string out_dir;
  std::vector<std::string> labels;   // 이번 판 이름 표(scenemap labels)
  OvdHandle* det = nullptr;
  sm_ctx* sm = nullptr;
  int64_t step = 0;
  double last_save = -1e9;
  int32_t n_kf = 0, n_det = 0;
  float det_ms = 0, save_ms = 0;
  float kf_ms = 0, crop_ms = 0;   // 마지막 keyframe: scenemap 갱신 전체(자르기 포함), best view 자르기(장치 → 호스트)
  int32_t n_crops = 0;
  int32_t n_png = 0;              // 마지막 저장에서 쓴 PNG 수
  int32_t n_ply = 0;
  float gather_ms = 0;            // 마지막 keyframe: 구름 점 색 모으기(장치 → 호스트)
  int32_t n_points = 0;
  sgrt_crop::Gpu* crop = nullptr; // 처음 장치 영상이 올 때 만듦
};

namespace {
void put(char* err, size_t n, const char* msg) {
  if (err && n) std::snprintf(err, n, "%s", msg);
}
double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
// sm_crop_fn: 이 keyframe 의 머리 RGB(장치 또는 호스트)에서 best view 상자만 자름
struct CropSrc {
  sgrt* s;
  const uint8_t* rgb;
  int64_t rs;
  int ps;
  int on_device;
  int w, h;
};
int cropCb(void* user, const sm_crop_req* reqs, int32_t n) {
  auto* c = static_cast<CropSrc*>(user);
  const auto t0 = std::chrono::steady_clock::now();
  int rc = 0;
  if (c->on_device) {
    if (!c->s->crop) c->s->crop = sgrt_crop::create();
    rc = c->s->crop ? sgrt_crop::run(c->s->crop, c->rgb, c->rs, c->ps, reqs, n) : -1;
  } else {
    for (int k = 0; k < n; ++k) scenemap::cropRgbHost(c->rgb, c->rs, c->ps, reqs[k]);
  }
  c->s->crop_ms += float(msSince(t0));
  c->s->n_crops += n;
  return rc;
}
// sm_gather_fn: 구름 점 색(남긴 화소만)
int gatherCb(void* user, const int32_t* xy, int32_t n, uint8_t* rgb) {
  auto* c = static_cast<CropSrc*>(user);
  const auto t0 = std::chrono::steady_clock::now();
  int rc = 0;
  if (c->on_device) {
    if (!c->s->crop) c->s->crop = sgrt_crop::create();
    rc = c->s->crop ? sgrt_crop::gather(c->s->crop, c->rgb, c->rs, c->ps, c->w, c->h, xy, n, rgb) : -1;
  } else {
    scenemap::gatherRgbHost(c->rgb, c->rs, c->ps, c->w, c->h, xy, n, rgb);
  }
  c->s->gather_ms += float(msSince(t0));
  c->s->n_points += n;
  return rc;
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
  sgrt_crop::destroy(s->crop);
  delete s;
}

int sgrt_begin(sgrt* s, const char* const* prompt, int32_t n, char* err, size_t err_len) {
  if (!s) return -1;
  // 프롬프트 방식: SGRT_PROMPT=task(과제 이름만) | all(엔진 어휘 전부) | auto(기본: 어휘가 kClosedVocabMax 이하인 닫힌 어휘
  // 엔진 — COCO-80 YOLO-seg — 이면 all, YOLOE 큰 어휘면 task)
  const char* mode = std::getenv("SGRT_PROMPT");
  const std::string m = mode ? mode : "auto";
  const int V = ovd_vocab_size(s->det);
  const bool all = m == "all" || (m != "task" && V <= kClosedVocabMax) || !prompt || n <= 0;
  const int found = ovd_set_prompt(s->det, prompt, n, err, err_len);   // 어휘 밖 이름은 err 에 적히고 번호는 유지(검출 안 됨)
  sm_reset(s->sm);
  s->labels.clear();
  if (all) {   // 엔진 어휘 전부(순서 = 엔진 번호). 과제 이름 중 어휘 밖의 것은 위 err 에 남음
    ovd_set_prompt(s->det, nullptr, 0, nullptr, 0);
    for (int i = 0; i < V; ++i) s->labels.push_back(ovd_vocab_name(s->det, i));
  } else {
    for (int i = 0; i < n; ++i) s->labels.push_back(prompt[i] ? prompt[i] : "");
  }
  std::vector<const char*> lp;
  for (const auto& l : s->labels) lp.push_back(l.c_str());
  sm_set_labels(s->sm, lp.data(), int(lp.size()));   // 같은 순서 = 검출 cls 가 그대로 이름 번호
  std::fprintf(stderr, "[sgrt] prompt %s: %d labels (task names in vocabulary %d/%d, engine vocabulary %d)\n", all ? "all" : "task",
               int(lp.size()), found, n, V);
  s->step = 0;
  s->last_save = -1e9;
  s->n_kf = s->n_det = 0;
  return 0;
}

int sgrt_set_kind_names(sgrt* s, int32_t kind, const char* const* names, int32_t n) {
  return s ? sm_set_kind_names(s->sm, kind, names, n) : -1;
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
    const auto t1 = std::chrono::steady_clock::now();
    s->crop_ms = s->gather_ms = 0;
    s->n_crops = s->n_points = 0;
    CropSrc cs{s, rgb, row_stride, pix_stride, rgb_on_device, w, h};
    const sm_rgb_source src{&cropCb, &gatherCb, &cs};
    rc = d ? sm_push_image_rgb(s->sm, &si, d, &src) : sm_push_image(s->sm, &si, nullptr);
    s->kf_ms = float(msSince(t1));
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
  sm_save_stats st{};
  const int rc = sm_save_dsg_ex(s->sm, s->out_dir.c_str(), &st);   // 바뀐 best view 만 PNG 로
  s->save_ms = float(msSince(t0));
  s->n_png = st.n_png;
  s->n_ply = st.n_ply;
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

void sgrt_get_timing(const sgrt* s, sgrt_timing* t) {
  if (!s || !t) return;
  *t = sgrt_timing{};
  t->det_ms = s->det_ms;
  t->kf_ms = s->kf_ms;
  t->crop_ms = s->crop_ms;
  t->save_ms = s->save_ms;
  t->n_crops = s->n_crops;
  t->n_png = s->n_png;
  t->n_ply = s->n_ply;
  t->gather_ms = s->gather_ms;
  t->n_points = s->n_points;
}

}  // extern "C"
