// G2-0: PhysX 없는 단독 실행기. 장면 파일 + 창 입력 흐름(core/scene/env_window.h, 대조기 G1_ENV_REC 가 씀)만으로 env 를 스텝하고
// 스텝마다 요약값(몸체·관절체·깸 카운터)이 흐름에 적힌 값(= 그 실행에서 PhysX 와 비트 같음이 확인된 우리 env)과 같은지 본다.
//   env_run <장면 파일> <창 입력 흐름> [--show N] [--envs N] [--threads T] [--batch]
// --envs N: 같은 장면 틀(SceneShared)을 나눠 쓰는 판 N 개에 같은 흐름을 넣어 스레드 T 개로 나눠 돌린다 — 판마다 요약값이 모두 같아야 한다
//           (판끼리 공유 전역 상태가 없음 = G2 N 판의 관문).
// --batch: 판들을 발맞춰 돌린다 (G2a 틀). 스텝마다 판마다 envStepBegin(판 짜기까지, 스레드 T 개) -> 모든 판의 풀이 본체를 한 번에
//          (runCore: 지금은 CPU — 판마다 solverStepHost+afterIntegrationHost 를 스레드로, GPU 판 sv::gpuSolveBatch 로 바꿀 자리) -> envStepEnd.
// 빌드: replay/CMakeLists.txt 의 env_run (ovd_replay_g1 과 같은 컴파일러·같은 부동소수 옵션, PhysX 링크 없음).
#include <xmmintrin.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/scene/env_load.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_window.h"

namespace sc2 = eng::scene;
namespace sv = eng::sv;

namespace {

struct EnvResult {
  uint64_t n = 0, badB = 0, badA = 0, badW = 0;
  long long firstBad = -1;
  double ms = 0;
  sc2::EnvTimes times;
  sc2::EnvWinStats st;
  uint64_t unknownEdge = 0, unknownNode = 0, engineErr = 0;
  std::string err;
  std::vector<std::string> shown;
  int show = 0;
};

// 판 하나 (적재된 env + 풀이 자리 + 쌍 관리층 앞 입력 표)
struct Env {
  std::unique_ptr<sc2::EnvOwned> o;
  std::unique_ptr<sc2::EnvSolveImpl> S;
  sc2::PairsStep front;
  bool load(const sc2::SceneFile& f, const sc2::SceneShared& sh, EnvResult& R) {
    o.reset(new sc2::EnvOwned);
    if (!sc2::envLoad(*o, f, sh, &R.err)) return false;
    S.reset(new sc2::EnvSolveImpl);
    S->load(f);
    S->seedPairs(o->C.S->pairs);
    o->E.solver = S.get();
    o->E.times = &R.times;
    return true;
  }
  void apply(const sc2::EnvWindow& W, EnvResult& R) { sc2::envApplyWindow(o->E, o->sc, *S, o->C.S->pairs, o->isl.M, W, front, R.st); }
  void check(const sc2::EnvWindow& W, EnvResult& R) {
    uint64_t b = 0, a = 0, w = 0;
    sc2::envDigest(o->E, *S, b, a, w);
    ++R.n;
    const bool okB = b == W.dBody, okA = a == W.dArt, okW = w == W.dWake;
    R.badB += !okB;
    R.badA += !okA;
    R.badW += !okW;
    if (!(okB && okA && okW)) {
      if (R.firstBad < 0) R.firstBad = (long long)W.sim;
      if (R.show > 0) {
        --R.show;
        char buf[160];
        snprintf(buf, sizeof(buf), "simulate %" PRIu64 ": 몸체 %s 관절체 %s 깸 카운터 %s", W.sim, okB ? "같음" : "다름", okA ? "같음" : "다름", okW ? "같음" : "다름");
        R.shown.push_back(buf);
      }
    }
  }
  void finish(EnvResult& R) {
    R.unknownEdge = S->unknownEdge;
    R.unknownNode = S->unknownNode;
    R.engineErr = S->engineErr;
  }
};

// 스레드 모음: parallelFor(n, fn) 은 fn(i) 를 i = 0..n-1 에 나눠 부르고 모두 끝나면 돌아온다.
// 나눔은 고정(스레드 w 가 i = w, w+T, ...) — 판이 늘 같은 스레드(같은 코어 캐시)에 머문다
struct Pool {
  // 도는 대기(spin): 발맞춰 모드는 스텝마다 세 번 모이므로 잠들었다 깨는 비용(스레드마다 수십 µs)이 크다. 부르는 스레드가 0 번 몫을 한다.
  std::vector<std::thread> th;
  const std::function<void(int)>* fn = nullptr;
  int n = 0, T = 1;
  std::atomic<uint64_t> gen{0};
  std::atomic<int> pending{0};
  std::atomic<bool> quit{false};
  explicit Pool(int t) : T(t) {
    for (int k = 1; k < t; ++k)
      th.emplace_back([this, k]() {
        uint64_t seen = 0;
        for (;;) {
          uint64_t g;
          for (int spin = 0; (g = gen.load(std::memory_order_acquire)) == seen; ++spin) {
            if (quit.load(std::memory_order_relaxed)) return;
            if (spin > 20000) std::this_thread::yield();
            else _mm_pause();
          }
          seen = g;
          for (int i = k; i < n; i += T) (*fn)(i);
          pending.fetch_sub(1, std::memory_order_acq_rel);
        }
      });
  }
  void parallelFor(int count, const std::function<void(int)>& f) {
    fn = &f;
    n = count;
    pending.store(T - 1, std::memory_order_relaxed);
    gen.fetch_add(1, std::memory_order_acq_rel);
    for (int i = 0; i < n; i += T) f(i);
    for (int spin = 0; pending.load(std::memory_order_acquire) != 0; ++spin)
      if (spin > 20000) std::this_thread::yield();
      else _mm_pause();
  }
  ~Pool() {
    quit.store(true);
    for (std::thread& t : th) t.join();
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: env_run <장면 파일> <창 입력 흐름> [--show N] [--envs N] [--threads T] [--batch]\n");
    return 2;
  }
  int show = 5, envs = 1, threads = 1;
  bool batchMode = false;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--batch")) batchMode = true;
    if (i + 1 >= argc) continue;
    if (!strcmp(argv[i], "--show")) show = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--envs")) envs = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
  }
  if (envs < 1) envs = 1;
  if (threads < 1) threads = 1;
  sc2::SceneFile f;
  std::string err;
  if (!sc2::readScene(argv[1], f, &err)) {
    fprintf(stderr, "장면 파일 못 읽음: %s\n", err.c_str());
    return 1;
  }
  std::unique_ptr<sc2::SceneShared> sh = sc2::makeShared(f);
  // 흐름 전체를 먼저 읽음 (편집·다시 맞춤 창 앞까지)
  FILE* in = fopen(argv[2], "rb");
  if (!in) {
    fprintf(stderr, "흐름 파일 못 엶: %s\n", argv[2]);
    return 1;
  }
  char magic[8];
  uint64_t s0 = 0;
  if (fread(magic, 1, 8, in) != 8 || memcmp(magic, sc2::kEnvWinMagic, 8) || fread(&s0, 8, 1, in) != 1) {
    fprintf(stderr, "흐름 머리가 다름\n");
    return 1;
  }
  if (s0 != f.h.sim) {
    fprintf(stderr, "흐름은 simulate %" PRIu64 " 파일에서, 장면 파일은 %" PRIu64 "\n", s0, uint64_t(f.h.sim));
    return 1;
  }
  std::vector<sc2::EnvWindow> wins;
  long long stopAt = -1;
  {
    sc2::EnvWindow W;
    while (sc2::envReadWindow(in, W)) {
      if (W.edit || W.unsup) {  // 편집 창·관절체 다시 맞춤: 흐름만으로 못 따라 함
        stopAt = (long long)W.sim;
        break;
      }
      wins.push_back(std::move(W));
    }
  }
  fclose(in);
  std::vector<EnvResult> R(static_cast<size_t>(envs));
  R[0].show = show;
  Pool pool(threads);
  double wall = 0, coreMs = 0;
  uint64_t coreCalls = 0;
  if (!batchMode) {  // 판마다 끝까지 따로
    const auto t0 = std::chrono::steady_clock::now();
    pool.parallelFor(envs, [&](int e) {
      Env V;
      EnvResult& r = R[size_t(e)];
      if (!V.load(f, *sh, r)) return;
      for (const sc2::EnvWindow& W : wins) {
        const auto t1 = std::chrono::steady_clock::now();
        V.apply(W, r);
        sc2::envStep(V.o->E);
        r.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        V.check(W, r);
      }
      V.finish(r);
    });
    wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  } else {  // 발맞춰: 판 짜기(스레드) -> 풀이 본체 한 번 -> 뒤 반쪽(스레드)
    std::vector<Env> V(static_cast<size_t>(envs));
    pool.parallelFor(envs, [&](int e) { V[size_t(e)].load(f, *sh, R[size_t(e)]); });
    for (int e = 0; e < envs; ++e)
      if (!R[size_t(e)].err.empty()) {
        fprintf(stderr, "env 적재 실패: %s\n", R[size_t(e)].err.c_str());
        return 1;
      }
    for (Env& v : V) v.S->coreExternal = true;
    std::vector<sv::SolverBoard*> boards(static_cast<size_t>(envs));
    std::vector<const sv::SolverParams*> prms(static_cast<size_t>(envs));
    for (int e = 0; e < envs; ++e) {
      boards[size_t(e)] = &V[size_t(e)].S->B;
      prms[size_t(e)] = &V[size_t(e)].S->prm;
    }
    // 풀이 본체 N 판 한 번 (지금 CPU: 판마다 스레드로. GPU 판이 나오면 sv::gpuSolveBatch(boards, prms, n))
    auto runCore = [&](sv::SolverBoard* const* B, const sv::SolverParams* const* P, int n) {
      pool.parallelFor(n, [&](int i) {
        sc2::EnvFtz ftz;
        sv::solverStepHost(*B[i], *P[i]);
        sv::afterIntegrationHost(*B[i]);
      });
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (const sc2::EnvWindow& W : wins) {
      pool.parallelFor(envs, [&](int e) {
        V[size_t(e)].apply(W, R[size_t(e)]);
        sc2::envStepBegin(V[size_t(e)].o->E);
      });
      const auto tc = std::chrono::steady_clock::now();
      runCore(boards.data(), prms.data(), envs);
      coreMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
      ++coreCalls;
      pool.parallelFor(envs, [&](int e) {
        sc2::envStepEnd(V[size_t(e)].o->E);
        V[size_t(e)].check(W, R[size_t(e)]);
      });
    }
    wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int e = 0; e < envs; ++e) V[size_t(e)].finish(R[size_t(e)]);
  }
  // 보고
  uint64_t badEnvs = 0, badB = 0, badA = 0, badW = 0, mism = 0;
  long long firstBad = -1;
  for (const EnvResult& r : R) {
    if (!r.err.empty()) {
      fprintf(stderr, "env 적재 실패: %s\n", r.err.c_str());
      return 1;
    }
    badB += r.badB;
    badA += r.badA;
    badW += r.badW;
    mism += r.st.mismatch;
    if (r.badB || r.badA || r.badW || r.st.mismatch || r.n != wins.size()) ++badEnvs;
    if (r.firstBad >= 0 && (firstBad < 0 || r.firstBad < firstBad)) firstBad = r.firstBad;
  }
  const EnvResult& r0 = R[0];
  for (const std::string& s : r0.shown) printf("  판 0 %s\n", s.c_str());
  printf("env_run (PhysX 없음, 장면 simulate %" PRIu64 "%s): 판 %d × 스텝 %zu — 요약값 다른 판 %" PRIu64 " (다름 몸체 %" PRIu64 " 관절체 %" PRIu64 " 깸 카운터 %" PRIu64 ")%s%s\n",
         uint64_t(f.h.sim), batchMode ? ", 발맞춰" : "", envs, wins.size(), badEnvs, badB, badA, badW,
         firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(firstBad)).c_str() : "",
         stopAt >= 0 ? ("  (simulate " + std::to_string(stopAt) + " 편집/다시 맞춤 창에서 멈춤)").c_str() : "");
  printf("  판 0 창: 바깥 섬 호출 %" PRIu64 ", 우리 쌍 호출 %" PRIu64 " (어긋남 %" PRIu64 "), 새 조인트 %" PRIu64 " 해제 %" PRIu64 "; 풀이: 모르는 간선 %" PRIu64 " 노드 %" PRIu64 " 판 오류 %" PRIu64 "\n",
         r0.st.ext, r0.st.ours, r0.st.mismatch, r0.st.jointsAdded, r0.st.jointsRemoved, r0.unknownEdge, r0.unknownNode, r0.engineErr);
  const double k = r0.n ? 1.0 / double(r0.n) : 0.0;
  const sc2::EnvTimes& tm = r0.times;
  if (!batchMode)
    printf("  판 0 시간: 창+스텝 %.3f ms/스텝 — 스텝 %.3f = 넓은 단계+쌍 %.3f, 좁은 단계 %.3f, 풀이 %.3f (그중 풀이 본체 %.3f), 적분 뒤 %.3f, 사라진 겹침 %.3f, 나머지(섬·순서기) %.3f\n",
           r0.ms * k, tm.total * k, tm.bp * k, tm.np * k, tm.solve * k, tm.solveCore * k, tm.after * k, tm.lost * k,
           (tm.total - tm.bp - tm.np - tm.solve - tm.after - tm.lost) * k);
  else
    printf("  발맞춰: 스텝마다 풀이 본체 한 번 (%" PRIu64 " 번) — 풀이 본체 %.3f ms/스텝(판 %d 개 합), 나머지 %.3f ms/스텝\n", coreCalls,
           coreCalls ? coreMs / double(coreCalls) : 0.0, envs, coreCalls ? (wall - coreMs) / double(coreCalls) : 0.0);
  if (envs > 1 || batchMode)
    printf("  전체: 판 %d 스레드 %d 벽시계 %.1f ms = 판·스텝당 %.4f ms (판 스텝 처리량 %.0f /s)\n", envs, threads, wall,
           wall / double(envs) / double(wins.size() ? wins.size() : 1), double(envs) * double(wins.size()) / (wall / 1000.0));
  return (badEnvs || mism) ? 3 : 0;
}
