// G2-0: PhysX 없는 단독 실행기. 장면 파일 + 창 입력 흐름(core/scene/env_window.h, 대조기 G1_ENV_REC 가 씀)만으로 env 를 스텝하고
// 스텝마다 요약값(몸체·관절체·깸 카운터)이 흐름에 적힌 값(= 그 실행에서 PhysX 와 비트 같음이 확인된 우리 env)과 같은지 본다.
//   env_run <장면 파일> <창 입력 흐름> [--show N] [--envs N] [--threads T]
// --envs N: 같은 장면 틀(SceneShared)을 나눠 쓰는 판 N 개에 같은 흐름을 넣어 스레드 T 개로 나눠 돌린다 — 판마다 요약값이 모두 같아야 한다
//           (판끼리 공유 전역 상태가 없음 = G2 N 판의 관문).
// --batch: 판마다 스레드 하나, 풀이 본체를 모든 판이 모여 한 번에(core/scene/env_batch.h — 지금은 CPU 흉내, 나중에 GPU).
// 빌드: replay/CMakeLists.txt 의 env_run (ovd_replay_g1 과 같은 컴파일러·같은 부동소수 옵션, PhysX 링크 없음).
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/scene/env_load.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_window.h"

namespace sc2 = eng::scene;

namespace {

struct EnvResult {
  uint64_t n = 0, badB = 0, badA = 0, badW = 0, mismatch = 0;
  long long firstBad = -1;
  double ms = 0;
  sc2::EnvTimes times;
  sc2::EnvWinStats st;
  uint64_t unknownEdge = 0, unknownNode = 0, engineErr = 0;
  std::string err;
  std::vector<std::string> shown;
};

// 판 하나: 적재 -> 창마다 넣고 스텝 -> 요약값 대조
void runEnv(const sc2::SceneFile& f, const sc2::SceneShared& sh, const std::vector<sc2::EnvWindow>& wins, int show, EnvResult& R, sc2::SolveBatch* batch) {
  struct Leave {
    sc2::SolveBatch* b;
    ~Leave() { if (b) b->leave(); }
  } leave{batch};
  std::unique_ptr<sc2::EnvOwned> o(new sc2::EnvOwned);
  if (!sc2::envLoad(*o, f, sh, &R.err)) return;
  std::unique_ptr<sc2::EnvSolveImpl> S(new sc2::EnvSolveImpl);
  S->batch = batch;
  S->load(f);
  S->seedPairs(o->C.S->pairs);
  o->E.solver = S.get();
  o->E.times = &R.times;
  sc2::PairsStep front;
  for (const sc2::EnvWindow& W : wins) {
    const auto t0 = std::chrono::steady_clock::now();
    sc2::envApplyWindow(o->E, o->sc, *S, o->C.S->pairs, o->isl.M, W, front, R.st);
    sc2::envStep(o->E);
    R.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint64_t b = 0, a = 0, w = 0;
    sc2::envDigest(o->E, *S, b, a, w);
    ++R.n;
    const bool okB = b == W.dBody, okA = a == W.dArt, okW = w == W.dWake;
    R.badB += !okB;
    R.badA += !okA;
    R.badW += !okW;
    if (!(okB && okA && okW)) {
      if (R.firstBad < 0) R.firstBad = (long long)W.sim;
      if (show > 0) {
        --show;
        char buf[160];
        snprintf(buf, sizeof(buf), "simulate %" PRIu64 ": 몸체 %s 관절체 %s 깸 카운터 %s", W.sim, okB ? "같음" : "다름", okA ? "같음" : "다름", okW ? "같음" : "다름");
        R.shown.push_back(buf);
      }
    }
  }
  R.unknownEdge = S->unknownEdge;
  R.unknownNode = S->unknownNode;
  R.engineErr = S->engineErr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: env_run <장면 파일> <창 입력 흐름> [--show N] [--envs N] [--threads T]\n");
    return 2;
  }
  int show = 5, envs = 1, threads = 1;
  bool batchMode = false;
  for (int i = 3; i < argc; ++i)
    if (!strcmp(argv[i], "--batch")) batchMode = true;
  for (int i = 3; i + 1 < argc; ++i) {
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
  // 판 N 개를 스레드 T 개로
  std::vector<EnvResult> R(static_cast<size_t>(envs));
  std::unique_ptr<sc2::SolveBatch> batch;
  if (batchMode) {
    threads = envs;  // 판마다 스레드 하나 (풀이 자리에서 모두 모임)
    batch.reset(new sc2::SolveBatch(envs));
    batch->run = [](eng::sv::SolverBoard* const* B, const eng::sv::SolverParams* const* P, int n) {  // CPU 흉내 (GPU 판이 나오면 교체)
      sc2::EnvFtz f;
      for (int i = 0; i < n; ++i) {
        eng::sv::solverStepHost(*B[i], *P[i]);
        eng::sv::afterIntegrationHost(*B[i]);
      }
    };
  }
  std::atomic<int> next{0};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&]() {
      for (int e; (e = next.fetch_add(1)) < envs;) runEnv(f, *sh, wins, e == 0 ? show : 0, R[size_t(e)], batch.get());
    });
  for (std::thread& th : pool) th.join();
  const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
  printf("env_run (PhysX 없음, 장면 simulate %" PRIu64 "): 판 %d × 스텝 %zu — 요약값 다른 판 %" PRIu64 " (다름 몸체 %" PRIu64 " 관절체 %" PRIu64 " 깸 카운터 %" PRIu64 ")%s%s\n",
         uint64_t(f.h.sim), envs, wins.size(), badEnvs, badB, badA, badW, firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(firstBad)).c_str() : "",
         stopAt >= 0 ? ("  (simulate " + std::to_string(stopAt) + " 편집/다시 맞춤 창에서 멈춤)").c_str() : "");
  printf("  판 0 창: 바깥 섬 호출 %" PRIu64 ", 우리 쌍 호출 %" PRIu64 " (어긋남 %" PRIu64 "), 새 조인트 %" PRIu64 " 해제 %" PRIu64 "; 풀이: 모르는 간선 %" PRIu64 " 노드 %" PRIu64 " 판 오류 %" PRIu64 "\n",
         r0.st.ext, r0.st.ours, r0.st.mismatch, r0.st.jointsAdded, r0.st.jointsRemoved, r0.unknownEdge, r0.unknownNode, r0.engineErr);
  const double k = r0.n ? 1.0 / double(r0.n) : 0.0;
  const sc2::EnvTimes& tm = r0.times;
  printf("  판 0 시간: 창+스텝 %.3f ms/스텝 — 스텝 %.3f = 넓은 단계+쌍 %.3f, 좁은 단계 %.3f, 풀이 %.3f (그중 풀이 본체 %.3f), 적분 뒤 %.3f, 사라진 겹침 %.3f, 나머지(섬·순서기) %.3f\n",
         r0.ms * k, tm.total * k, tm.bp * k, tm.np * k, tm.solve * k, tm.solveCore * k, tm.after * k, tm.lost * k,
         (tm.total - tm.bp - tm.np - tm.solve - tm.after - tm.lost) * k);
  if (batch) printf("  풀이 모으기: 실행 %" PRIu64 " 번, 판 %" PRIu64 " 개 (실행당 평균 %.1f 판)\n", batch->launches, batch->boardsSolved,
                    batch->launches ? double(batch->boardsSolved) / double(batch->launches) : 0.0);
  if (envs > 1)
    printf("  전체: 판 %d 스레드 %d 벽시계 %.1f ms = 판·스텝당 %.4f ms (판 스텝 처리량 %.0f /s)\n", envs, threads, wall,
           wall / double(envs) / double(wins.size() ? wins.size() : 1), double(envs) * double(wins.size()) / (wall / 1000.0));
  return (badEnvs || mism) ? 3 : 0;
}
