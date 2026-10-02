// G2 판 N 개 실행 도구 (env_run·native_loop 공용, 호스트): 판 하나(Env: 적재·창 넣기·요약값 대조), 스레드 모음(Pool: 도는 대기, 고정 나눔).
#pragma once
#include <xmmintrin.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/scene/env_load.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_window.h"

namespace envrun {
namespace sc2 = eng::scene;
namespace sv = eng::sv;

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


// 흐름 파일 읽기 (편집·다시 맞춤 창 앞까지). stopAt: 멈춘 창 simulate (-1 = 끝까지)
inline bool readStream(const char* path, uint64_t sceneSim, std::vector<sc2::EnvWindow>& wins, long long& stopAt, std::string& err) {
  FILE* in = fopen(path, "rb");
  if (!in) {
    err = std::string("흐름 파일 못 엶: ") + path;
    return false;
  }
  char magic[8];
  uint64_t s0 = 0;
  if (fread(magic, 1, 8, in) != 8 || memcmp(magic, sc2::kEnvWinMagic, 8) || fread(&s0, 8, 1, in) != 1) {
    fclose(in);
    err = "흐름 머리가 다름";
    return false;
  }
  if (s0 != sceneSim) {
    fclose(in);
    err = "흐름은 simulate " + std::to_string(s0) + " 파일에서, 장면 파일은 " + std::to_string(sceneSim);
    return false;
  }
  stopAt = -1;
  sc2::EnvWindow W;
  while (sc2::envReadWindow(in, W)) {
    if (W.edit || W.unsup) {
      stopAt = (long long)W.sim;
      break;
    }
    wins.push_back(std::move(W));
  }
  fclose(in);
  return true;
}

}  // namespace envrun
