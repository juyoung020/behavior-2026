// N2 네이티브 루프 (G3 단계): 판 N 개 물리(장면 파일 + 창 입력 흐름, 발맞춰) + 렌더(RenderBatch, 판 N × 카메라 3) — 파이썬 없음.
//   native_loop <장면 파일> <창 입력 흐름> <렌더 폴더 | -> [--envs N] [--threads T] [--render-every K] [--steps S] [--ctrl <기록 폴더>]
// --ctrl: 로봇 드라이브 목표를 흐름 대신 우리 제어기(native_ctrl.h, 기록 행동 s1_actions.bin)로 — 흐름 값과 비교, 요약값도 같아야 함.
// 렌더 폴더가 - 이면 렌더 없이 (물리·제어만).
// --free (--ctrl 과 함께): 경계 뒤 창에서 흐름 입력(섬 호출·깸 카운터/활성 차이·쌍 관리층 표)을 하나도 안 쓰고, 로봇 입력만 엔진 API 로 —
//   드라이브 목표(우리 제어기) = setDriveTarget/Velocity(autowake: 깸 카운터 < 0.4 면 NpArticulation autoWakeInternal = EnvArtApi::wakeUpInternal),
//   흐름에 로봇 wakeUp/putToSleep 호출이 있던 자리 = EnvArtApi::wakeUp/putToSleep (언제 부르는지는 아직 흐름에서).
// 렌더 폴더 = scene.rsc + frame_<k>.rfr (render_capture -> convert_scene) + anchors.txt (export_render_anchors.py).
// 확인:
//  (1) 물리 = 흐름 요약값 (판마다, env_run 과 같음)
//  (2) 기준 prim 짝짓기: 판 0 의 우리 기준 prim 행렬(물리 상태 -> 행위자 전역 자세 -> 행렬)을 렌더 캡처 프레임의 행렬과 대조.
//      프레임 k 와 가장 가까운 simulate 를 찾아 움직이는 기준 prim 들의 최대 차를 적는다 (캡처가 같은 판·같은 행동이면 작아야 함).
//  (3) 시간: 물리(판 짜기·풀이·뒤 반쪽)·행렬 만들기·렌더(GPU) 스텝당.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "core/scene/env_art_api.h"
#include "core/scene/env_render.h"
#include "tests/scene/env_lockstep.h"
#include "tests/scene/native_ctrl.h"
#include "tests/scene/native_obs.h"
#include "tests/scene/native_render.h"

using namespace envrun;
namespace rnd = eng::rnd;

namespace {

static const char* kRoles[3] = {"head", "left_wrist", "right_wrist"};

// 카메라마다 프레임 사이 상대 변환이 가장 일정한 기준 prim 을 찾아 CamRig 로 (tests/render/test_render_scene.cu 4) 와 같은 방법, double)
std::vector<rnd::CamRig> rigsFromFrames(const std::vector<rnd::HostFrame>& frames, int nAnchor) {
  auto inv_d = [](const rnd::Aff& a, double* o) {
    double m[12];
    for (int k = 0; k < 12; ++k) m[k] = a.m[k];
    const double c00 = m[5] * m[10] - m[6] * m[9], c01 = m[6] * m[8] - m[4] * m[10], c02 = m[4] * m[9] - m[5] * m[8];
    const double id = 1.0 / (m[0] * c00 + m[1] * c01 + m[2] * c02);
    o[0] = c00 * id; o[1] = (m[2] * m[9] - m[1] * m[10]) * id; o[2] = (m[1] * m[6] - m[2] * m[5]) * id;
    o[4] = c01 * id; o[5] = (m[0] * m[10] - m[2] * m[8]) * id; o[6] = (m[2] * m[4] - m[0] * m[6]) * id;
    o[8] = c02 * id; o[9] = (m[1] * m[8] - m[0] * m[9]) * id; o[10] = (m[0] * m[5] - m[1] * m[4]) * id;
    for (int r = 0; r < 3; ++r) o[r * 4 + 3] = -(o[r * 4] * m[3] + o[r * 4 + 1] * m[7] + o[r * 4 + 2] * m[11]);
  };
  auto mul_d = [](const double* a, const rnd::Aff& b, double* o) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) o[i * 4 + j] = a[i * 4] * b.m[j] + a[i * 4 + 1] * b.m[4 + j] + a[i * 4 + 2] * b.m[8 + j];
      o[i * 4 + 3] += a[i * 4 + 3];
    }
  };
  const int nf = int(frames.size());
  std::vector<rnd::CamRig> rigs(3);
  for (int c = 0; c < 3; ++c) {
    int best = -1;
    double best_dev = 1e30, rel0[12] = {0};
    for (int a = 0; a < nAnchor; ++a) {
      double r0[12], dev = 0;
      for (int f = 0; f < nf; ++f) {
        double iv[12], r[12];
        inv_d(frames[f].anchor[a], iv);
        mul_d(iv, frames[f].cams[c].world, r);
        if (f == 0) memcpy(r0, r, sizeof r0);
        for (int k = 0; k < 12; ++k) dev = std::max(dev, std::fabs(r[k] - r0[k]));
      }
      if (dev < best_dev) {
        best_dev = dev;
        best = a;
        memcpy(rel0, r0, sizeof rel0);
      }
    }
    rnd::CamRig& g = rigs[c];
    g.anchor = best;
    for (int k = 0; k < 12; ++k) g.rel.m[k] = float(rel0[k]);
    const rnd::Camera& c0 = frames[0].cams[c];
    g.w = c0.w;
    g.h = c0.h;
    g.tanx = c0.tanx;
    g.tany = c0.tany;
    g.znear = c0.znear;
    g.zfar = c0.zfar;
    printf("  카메라 %-11s %dx%d 기준 prim %d 에 붙음 (프레임 사이 rel 흔들림 %.2e)\n", kRoles[c], g.w, g.h, best, best_dev);
  }
  return rigs;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: native_loop <장면 파일> <창 입력 흐름> <렌더 폴더> [--envs N] [--threads T] [--render-every K] [--steps S]\n");
    return 2;
  }
  int envs = 1, threads = 1, renderEvery = 1;
  long long maxSteps = -1;
  std::string ctrlDir;
  bool freeRun = false;
  for (int i = 4; i < argc; ++i)
    if (!strcmp(argv[i], "--free")) freeRun = true;
  for (int i = 4; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--envs")) envs = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--render-every")) renderEvery = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--steps")) maxSteps = atoll(argv[i + 1]);
    if (!strcmp(argv[i], "--ctrl")) ctrlDir = argv[i + 1];
  }
  envs = std::max(envs, 1);
  threads = std::max(threads, 1);
  // ---- 물리
  sc2::SceneFile f;
  std::string err;
  if (!sc2::readScene(argv[1], f, &err)) {
    fprintf(stderr, "장면 파일 못 읽음: %s\n", err.c_str());
    return 1;
  }
  std::unique_ptr<sc2::SceneShared> sh = sc2::makeShared(f);
  std::vector<sc2::EnvWindow> wins;
  long long stopAt = -1;
  if (!readStream(argv[2], f.h.sim, wins, stopAt, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  if (maxSteps >= 0 && size_t(maxSteps) < wins.size()) wins.resize(size_t(maxSteps));
  // ---- 렌더 장면·프레임·기준 prim 경로
  const std::string rdir = argv[3];
  const bool doRender = rdir != "-";
  rnd::HostScene H;
  if (doRender && !rnd::load_scene(rdir + "/scene.rsc", H)) {
    fprintf(stderr, "렌더 장면 못 읽음\n");
    return 1;
  }
  std::vector<std::string> fnames;
  if (DIR* d = doRender ? opendir(rdir.c_str()) : nullptr) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 10 && n.compare(0, 6, "frame_") == 0 && n.compare(n.size() - 4, 4, ".rfr") == 0) fnames.push_back(n);
    }
    closedir(d);
  }
  std::sort(fnames.begin(), fnames.end());
  std::vector<rnd::HostFrame> frames;
  for (const std::string& n : fnames) {
    rnd::HostFrame F;
    if (rnd::load_frame(rdir + "/" + n, F)) frames.push_back(std::move(F));
  }
  if (doRender && frames.empty()) {
    fprintf(stderr, "렌더 프레임이 없음\n");
    return 1;
  }
  std::vector<std::string> paths;
  {
    std::ifstream in(rdir + "/anchors.txt");
    for (std::string l; std::getline(in, l);)
      if (!l.empty()) paths.push_back(l);
  }
  sc2::EnvRenderMap map;
  if (doRender && !map.build(f, paths, frames[0].anchor, &err)) {
    fprintf(stderr, "기준 prim 짝짓기 실패: %s\n", err.c_str());
    return 1;
  }
  const size_t A = map.a.size();
  printf("native_loop: 판 %d, 스레드 %d, 스텝 %zu%s | 렌더 기준 prim %zu (몸체 %u, 링크 %u, 정적 행위자 %u, 고정 %u), 삼각형 %zu, 프레임 %zu\n", envs, threads,
         wins.size(), stopAt >= 0 ? " (편집 창 앞에서 멈춤)" : "", A, map.n[1], map.n[2], map.n[3], map.n[0], H.tris.size(), frames.size());
  nrender::NativeRender* nr = nullptr;
  if (doRender) nr = nrender::create(H, envs, rigsFromFrames(frames, int(A)), frames[0].vis);
  // ---- 판 N 개 발맞춰
  std::vector<EnvResult> R(static_cast<size_t>(envs));
  R[0].show = 5;
  Pool pool(threads);
  std::vector<Env> V(static_cast<size_t>(envs));
  pool.parallelFor(envs, [&](int e) { V[size_t(e)].load(f, *sh, R[size_t(e)]); });
  for (int e = 0; e < envs; ++e)
    if (!R[size_t(e)].err.empty()) {
      fprintf(stderr, "env 적재 실패: %s\n", R[size_t(e)].err.c_str());
      return 1;
    }
  for (Env& v : V) v.S->coreExternal = true;
  // 로봇 제어기 (--ctrl)
  nctrl::Setup CS;
  std::vector<nctrl::Ctrl> ctl(static_cast<size_t>(envs));
  std::vector<float> acts;
  uint32_t actT = 0;
  if (!ctrlDir.empty()) {
    if (!CS.load(ctrlDir, err) || !CS.bind(f, *V[0].S, err)) {
      fprintf(stderr, "제어기: %s\n", err.c_str());
      return 1;
    }
    FILE* fa = fopen((ctrlDir + "/s1_actions.bin").c_str(), "rb");
    uint32_t A2 = 0;
    if (!fa || fread(&actT, 4, 1, fa) != 1 || fread(&A2, 4, 1, fa) != 1 || A2 != 23) {
      fprintf(stderr, "s1_actions.bin 을 못 읽음\n");
      return 1;
    }
    acts.resize(size_t(actT) * 23);
    if (fread(acts.data(), 4, acts.size(), fa) != acts.size()) return 1;
    fclose(fa);
    printf("  제어기: 로봇 관절체 %u, dof %d, 행동 %u 스텝, 에피소드 시작 simulate %llu\n", CS.art, CS.n_dof, actT, (unsigned long long)CS.episode_start);
  }
  std::vector<sc2::EnvWindow> wctl(static_cast<size_t>(envs));
  // 관측 proprio (기록 폴더에 obs_setup.txt·obs_ref.bin 이 있으면): 스텝 끝마다 판 0 을 공식 trace 와 비교
  nobs::Setup OS;
  std::vector<float> obsRef;
  uint32_t obsT = 0;
  uint64_t obsSteps = 0, obsBadSteps = 0, obsBadFields = 0;
  long long obsFirstBad = -1;
  if (CS.bound && OS.load(ctrlDir, err) && OS.bind(f, CS, err)) {
    FILE* fo = fopen((ctrlDir + "/obs_ref.bin").c_str(), "rb");
    int32_t hdr[2] = {0, 0};
    if (fo && fread(hdr, 4, 2, fo) == 2 && hdr[1] == 61) {
      obsT = uint32_t(hdr[0]);
      obsRef.resize(size_t(obsT) * 61);
      if (fread(obsRef.data(), 4, obsRef.size(), fo) != obsRef.size()) obsT = 0;
    }
    if (fo) fclose(fo);
    printf("  관측: proprio 61 기준 %u 스텝 (MKL %s)\n", obsT, getenv("ENGINE_MKL_LIB") ? "있음" : "없음 — cos/sin 이 다를 수 있음");
  }
  if (freeRun && !CS.bound) {
    fprintf(stderr, "--free 는 --ctrl 과 함께\n");
    return 1;
  }
  std::vector<std::unique_ptr<sc2::EnvArtApi>> artApi(static_cast<size_t>(envs));
  for (int e = 0; e < envs; ++e) artApi[size_t(e)].reset(new sc2::EnvArtApi(V[size_t(e)].o->E, *V[size_t(e)].S));
  std::vector<uint64_t> freeOther(static_cast<size_t>(envs), 0);
  // 흐름 없는 창: 로봇 호출만 엔진 API 로
  auto applyFree = [&](int e, const sc2::EnvWindow& W, const sc2::EnvWindow& Wc) {
    sc2::EnvWindow Wf;
    Wf.sim = W.sim;
    V[size_t(e)].apply(Wf, R[size_t(e)]);
    sc2::EnvArtApi& api = *artApi[size_t(e)];
    eng::art::Articulation& Ar = V[size_t(e)].S->arts[CS.art];
    const uint32_t node = api.nodeOf(int32_t(CS.art));
    for (const sc2::EnvArtOp& op : Wc.artOps) {
      if (op.art != CS.art) {
        ++freeOther[size_t(e)];
        continue;
      }
      if (op.type == 2) api.wakeUp(int32_t(op.art));
      else if (op.type == 3) api.putToSleep(int32_t(op.art));
      else {
        if (Ar.wakeCounter < sc2::kWakeReset) api.wakeUpInternal(Ar, node, false);  // autoWakeInternal
        if (op.type == 0) eng::art::jointSetDriveTarget(Ar, op.link, op.axis, op.v, false);
        else eng::art::jointSetDriveVelocity(Ar, op.link, op.axis, op.v, false);
      }
    }
    // 진단 (NL_FREE_DIAG=n): 흐름이 적은 창 뒤 깸 카운터·활성 값과 우리 API 결과가 다른 칸 (앞 n 창)
    static int diag = getenv("NL_FREE_DIAG") ? atoi(getenv("NL_FREE_DIAG")) : 0;
    if (e == 0 && diag > 0) {
      --diag;
      sc2::EnvStep& E = V[0].o->E;
      int shown = 0;
      for (const sc2::HostBodyWake& w : W.wakeBodies) {
        const sc2::HostBodyWake* q = E.wake.body(w.node);
        if (q && !memcmp(q, &w, sizeof(w))) continue;
        if (shown++ < 6)
          printf("    [free 진단 sim %llu] 몸체/링크 노드 %llx 흐름 wc %.9g solverWc %.9g solveWc %.9g / 우리 %.9g %.9g %.9g\n", (unsigned long long)W.sim,
                 (unsigned long long)w.node, w.wc, w.solverWc, w.solveWc, q ? q->wc : -1.f, q ? q->solverWc : -1.f, q ? q->solveWc : -1.f);
      }
      for (const sc2::HostArtWake& w : W.wakeArts) {
        const sc2::HostArtWake* q = E.wake.art(w.node);
        if (q && q->wc == w.wc) continue;
        if (shown++ < 12) printf("    [free 진단 sim %llu] 관절체 노드 %u 흐름 wc %.9g / 우리 %.9g\n", (unsigned long long)W.sim, w.node, w.wc, q ? q->wc : -1.f);
      }
      for (const auto& x : W.active) {
        if (x.first < E.active.size() && E.active[x.first] == x.second) continue;
        if (shown++ < 18) printf("    [free 진단 sim %llu] 행위자 %u 활성 흐름 %u / 우리 %d\n", (unsigned long long)W.sim, x.first, x.second,
                                 x.first < E.active.size() ? int(E.active[x.first]) : -1);
      }
      printf("    [free 진단 sim %llu] 흐름 차이 칸 몸체 %zu 관절체 %zu 활성 %zu, 그중 우리와 다른 칸 %d\n", (unsigned long long)W.sim, W.wakeBodies.size(),
             W.wakeArts.size(), W.active.size(), shown);
    }
  };
  std::vector<rnd::Aff> anch(size_t(envs) * A);
  // 기준 prim 대조: 프레임마다 가장 가까운 simulate (판 0, 움직이는 기준 prim = 몸체·링크)
  std::vector<double> bestDiff(frames.size(), 1e30);
  std::vector<long long> bestSim(frames.size(), -1);
  auto checkAnchors = [&](uint64_t sim) {
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      double m = 0;
      for (size_t k = 0; k < A; ++k) {
        const uint8_t kd = map.a[k].kind;
        if (kd != sc2::EnvRenderAnchor::kBody && kd != sc2::EnvRenderAnchor::kLink) continue;
        for (int q = 0; q < 12; ++q) m = std::max(m, double(std::fabs(anch[k].m[q] - frames[fi].anchor[k].m[q])));
      }
      if (m < bestDiff[fi]) {
        bestDiff[fi] = m;
        bestSim[fi] = (long long)sim;
      }
    }
  };
  double msBegin = 0, msCore = 0, msEnd = 0, msAnch = 0, msRender = 0;
  uint64_t renders = 0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto tAll = now();
  for (size_t w = 0; w < wins.size(); ++w) {
    const sc2::EnvWindow& W = wins[w];
    auto t0 = now();
    pool.parallelFor(envs, [&](int e) {
      const float* A_ = acts.data();
      const bool c = CS.bound && ctl[size_t(e)].apply(CS, *V[size_t(e)].S, W, A_, actT, wctl[size_t(e)]);
      if (freeRun && !W.first)
        applyFree(e, W, c ? wctl[size_t(e)] : W);
      else
        V[size_t(e)].apply(c ? wctl[size_t(e)] : W, R[size_t(e)]);
      sc2::envStepBegin(V[size_t(e)].o->E);
    });
    auto t1 = now();
    pool.parallelFor(envs, [&](int e) {  // 풀이 본체 (CPU. gpuSolveBatch 가 오면 한 번에)
      sc2::EnvFtz ftz;
      sv::solverStepHost(V[size_t(e)].S->B, V[size_t(e)].S->prm);
      sv::afterIntegrationHost(V[size_t(e)].S->B);
    });
    auto t2 = now();
    pool.parallelFor(envs, [&](int e) {
      sc2::envStepEnd(V[size_t(e)].o->E);
      V[size_t(e)].check(W, R[size_t(e)]);
    });
    auto t3 = now();
    if (OS.bound && obsT && W.sim > CS.episode_start && (W.sim - CS.episode_start) % CS.substeps == 0) {  // 스텝 t 의 마지막 서브스텝 뒤
      const uint64_t t = (W.sim - CS.episode_start) / CS.substeps - 1;
      if (t < obsT) {
        float o[61];
        OS.proprio(CS, *V[0].S, o);
        int bad = 0;
        for (int i = 0; i < 61; ++i) bad += memcmp(&o[i], &obsRef[size_t(t) * 61 + size_t(i)], 4) != 0;
        ++obsSteps;
        if (bad) {
          ++obsBadSteps;
          obsBadFields += uint64_t(bad);
          if (obsFirstBad < 0) {
            obsFirstBad = (long long)t;
            for (int i = 0; i < 61; ++i)
              if (memcmp(&o[i], &obsRef[size_t(t) * 61 + size_t(i)], 4))
                printf("    [관측 스텝 %llu] 칸 %d 우리 %.9g 공식 %.9g\n", (unsigned long long)t, i, o[i], obsRef[size_t(t) * 61 + size_t(i)]);
          }
        }
      }
    }
    msBegin += ms(t0, t1);
    msCore += ms(t1, t2);
    msEnd += ms(t2, t3);
    // 스텝 뒤 상태 -> 기준 prim 행렬 (다음 스텝 관측, k-1 규칙) -> 렌더
    pool.parallelFor(envs, [&](int e) { map.anchors(*V[size_t(e)].S, anch.data() + size_t(e) * A); });
    auto t4 = now();
    msAnch += ms(t3, t4);
    if (!doRender) continue;
    checkAnchors(W.sim);
    if (renderEvery > 0 && (w % size_t(renderEvery)) == 0) {
      nrender::setAnchors(nr, anch.data(), anch.size());
      msRender += nrender::render(nr, int(w));
      ++renders;
    }
  }
  const double wall = ms(tAll, now());
  // ---- 보고
  uint64_t badEnvs = 0;
  for (const EnvResult& r : R)
    if (r.badB || r.badA || r.badW || r.st.mismatch || r.n != wins.size()) ++badEnvs;
  printf("  물리: 판 %d × 스텝 %zu 요약값 다른 판 %" PRIu64 "\n", envs, wins.size(), badEnvs);
  for (const std::string& sh0 : R[0].shown) printf("    판 0 %s\n", sh0.c_str());
  if (CS.bound)
    printf("  제어기 (판 0): 행동 스텝 %" PRIu64 ", 드라이브 목표 흐름과 비교 %" PRIu64 " 다름 %" PRIu64 "%s\n", ctl[0].steps, ctl[0].cmpN, ctl[0].cmpBad,
           ctl[0].firstBad >= 0 ? (" 첫 다름 simulate " + std::to_string(ctl[0].firstBad)).c_str() : "");
  if (obsSteps)
    printf("  관측 proprio 61 (판 0): 스텝 %" PRIu64 " 중 다른 스텝 %" PRIu64 " (칸 %" PRIu64 ")%s\n", obsSteps, obsBadSteps, obsBadFields,
           obsFirstBad >= 0 ? (" 첫 다름 스텝 " + std::to_string(obsFirstBad)).c_str() : "");
  if (freeRun) printf("  --free: 흐름 입력 없이 (판 0 로봇 아닌 관절체 호출 빠뜨림 %" PRIu64 ", EnvArtApi 깨움 요청 %" PRIu64 ")\n", freeOther[0], artApi[0]->reqActivate);
  if (doRender) printf("  기준 prim 대조 (판 0, 몸체·링크 %u 개, 렌더 캡처 프레임별 가장 가까운 simulate):\n", map.n[1] + map.n[2]);
  for (size_t fi = 0; fi < frames.size() && doRender; ++fi)
    printf("    프레임 스텝 %4" PRId64 ": simulate %lld 에서 최대 차 %.3g\n", frames[fi].step, bestSim[fi], bestDiff[fi]);
  const double n = double(wins.size());
  printf("  시간 ms/스텝: 판 짜기 %.3f, 풀이 본체(CPU) %.3f, 뒤 반쪽 %.3f, 기준 prim 행렬 %.3f, 렌더(GPU, 판 %d × 카메라 3) %.3f (그린 스텝 %" PRIu64 ")\n",
         msBegin / n, msCore / n, msEnd / n, msAnch / n, renders ? msRender / double(renders) : 0.0, envs, renders);
  printf("  전체 벽시계 %.1f ms = 판·스텝당 %.4f ms (판 스텝 처리량 %.0f /s)\n", wall, wall / n / envs, n * envs / (wall / 1000.0));
  // 판 0 첫/끝 카메라 영상 일부 값 (눈으로 보기용 저장은 다음)
  nrender::destroy(nr);
  return badEnvs ? 3 : 0;
}
