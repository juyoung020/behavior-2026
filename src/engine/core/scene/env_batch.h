// G2a 풀이 모으기: 판 N 개가 각자 스레드에서 envStep 을 돌다가 풀이 본체(solverStep + afterIntegration) 자리에서 만나
// 한 번에 푼다. 모두 모이면 마지막에 온 스레드가 run(판 목록)을 부르고 나머지를 깨운다.
//   run 기본값 = CPU 흉내(판마다 solverStepHost + afterIntegrationHost, FTZ) — GPU 판(sv::gpuSolveBatch, engine-solver-art)이 나오면 그것으로 바꾼다.
// 판끼리는 독립이라 모이는 차례(도착 순서)가 결과에 영향이 없다.
// 판이 끝나면(더 스텝하지 않으면) leave() 로 참가자에서 빠진다.
#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "core/solver/solver_io.h"

namespace eng {
namespace scene {

struct SolveBatch {
  using Run = std::function<void(sv::SolverBoard* const* boards, const sv::SolverParams* const* prms, int n)>;
  std::mutex m;
  std::condition_variable cv;
  int participants = 0, arrived = 0;
  uint64_t gen = 0, launches = 0, boardsSolved = 0;
  std::vector<sv::SolverBoard*> boards;
  std::vector<const sv::SolverParams*> prms;
  Run run;

  explicit SolveBatch(int n) : participants(n) {}

  void solve(sv::SolverBoard& B, const sv::SolverParams& p) {
    std::unique_lock<std::mutex> l(m);
    boards.push_back(&B);
    prms.push_back(&p);
    ++arrived;
    if (arrived >= participants) {
      launch();
      return;
    }
    const uint64_t g = gen;
    cv.wait(l, [&] { return gen != g; });
  }
  void leave() {
    std::unique_lock<std::mutex> l(m);
    --participants;
    if (arrived > 0 && arrived >= participants) launch();
  }

 private:
  void launch() {  // 잠금 안에서
    run(boards.data(), prms.data(), int(boards.size()));
    ++launches;
    boardsSolved += boards.size();
    boards.clear();
    prms.clear();
    arrived = 0;
    ++gen;
    cv.notify_all();
  }
};

}  // namespace scene
}  // namespace eng
