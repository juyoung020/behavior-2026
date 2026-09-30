// G2-0: PhysX 없는 단독 실행기. 장면 파일 + 창 입력 흐름(core/scene/env_window.h, 대조기 G1_ENV_REC 가 씀)만으로 env 를 스텝하고
// 스텝마다 요약값(몸체·관절체·깸 카운터)이 흐름에 적힌 값(= 그 실행에서 PhysX 와 비트 같음이 확인된 우리 env)과 같은지 본다.
//   env_run <장면 파일> <창 입력 흐름> [--show N]
// 빌드: replay/CMakeLists.txt 의 env_run (ovd_replay_g1 과 같은 컴파일러·같은 부동소수 옵션, PhysX 링크 없음).
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "core/scene/env_load.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_window.h"

namespace sc2 = eng::scene;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: env_run <장면 파일> <창 입력 흐름> [--show N]\n");
    return 2;
  }
  int show = 5;
  for (int i = 3; i + 1 < argc; ++i)
    if (!strcmp(argv[i], "--show")) show = atoi(argv[i + 1]);
  sc2::SceneFile f;
  std::string err;
  if (!sc2::readScene(argv[1], f, &err)) {
    fprintf(stderr, "장면 파일 못 읽음: %s\n", err.c_str());
    return 1;
  }
  std::unique_ptr<sc2::SceneShared> sh = sc2::makeShared(f);
  std::unique_ptr<sc2::EnvOwned> o(new sc2::EnvOwned);
  if (!sc2::envLoad(*o, f, *sh, &err)) {
    fprintf(stderr, "env 적재 실패: %s\n", err.c_str());
    return 1;
  }
  std::unique_ptr<sc2::EnvSolveImpl> S(new sc2::EnvSolveImpl);
  S->load(f);
  S->seedPairs(o->C.S->pairs);
  o->E.solver = S.get();
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
  sc2::PairsStep front;
  sc2::EnvWindow W;
  sc2::EnvWinStats st;
  uint64_t n = 0, badB = 0, badA = 0, badW = 0;
  long long firstBad = -1, stopAt = -1;
  double ms = 0;
  while (sc2::envReadWindow(in, W)) {
    if (W.edit || W.unsup) {  // 편집 창·관절체 다시 맞춤: 흐름만으로 못 따라 함
      stopAt = (long long)W.sim;
      break;
    }
    const auto t0 = std::chrono::steady_clock::now();
    sc2::envApplyWindow(o->E, o->sc, *S, o->C.S->pairs, o->isl.M, W, front, st);
    sc2::envStep(o->E);
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint64_t b = 0, a = 0, w = 0;
    sc2::envDigest(o->E, *S, b, a, w);
    ++n;
    const bool okB = b == W.dBody, okA = a == W.dArt, okW = w == W.dWake;
    badB += !okB;
    badA += !okA;
    badW += !okW;
    if (!(okB && okA && okW)) {
      if (firstBad < 0) firstBad = (long long)W.sim;
      if (show > 0) {
        --show;
        printf("  simulate %" PRIu64 ": 몸체 %s 관절체 %s 깸 카운터 %s\n", W.sim, okB ? "같음" : "다름", okA ? "같음" : "다름", okW ? "같음" : "다름");
      }
    }
  }
  fclose(in);
  printf("env_run (PhysX 없음, 장면 simulate %" PRIu64 "): 스텝 %" PRIu64 " — 요약값 다름 몸체 %" PRIu64 " 관절체 %" PRIu64 " 깸 카운터 %" PRIu64 "%s%s\n", uint64_t(f.h.sim), n, badB,
         badA, badW, firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(firstBad)).c_str() : "",
         stopAt >= 0 ? ("  (simulate " + std::to_string(stopAt) + " 편집/다시 맞춤 창에서 멈춤)").c_str() : "");
  printf("  창: 바깥 섬 호출 %" PRIu64 ", 우리 쌍 호출 %" PRIu64 " (어긋남 %" PRIu64 "), 새 조인트 %" PRIu64 " 해제 %" PRIu64 "; 풀이: 모르는 간선 %" PRIu64 " 노드 %" PRIu64 " 판 오류 %" PRIu64 "\n",
         st.ext, st.ours, st.mismatch, st.jointsAdded, st.jointsRemoved, S->unknownEdge, S->unknownNode, S->engineErr);
  printf("  시간: 창+스텝 %.3f ms/스텝 (CPU 한 판)\n", n ? ms / double(n) : 0.0);
  return (badB || badA || badW || st.mismatch) ? 3 : 0;
}
