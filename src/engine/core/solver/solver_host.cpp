// solver 호스트 진입 함수 (층 1). solver_io.h 에 선언, 풀이 본체(tgs_solver.h, aos 사용)를 이 번역 단위에만 넣는다.
// 그래서 PhysX 헤더를 쓰는 쪽(통합 비계·정답지 시험)은 solver_io.h 만 include 하고 이 파일을 함께 빌드하면 된다
// (common/aos.h 의 SSE 흉내 이름이 PhysX 쪽 xmmintrin.h 매크로와 한 번역 단위에서 부딪히지 않게).
#include "tgs_solver.h"

namespace eng {
namespace sv {

void solverStepHost(SolverBoard& B, const SolverParams& prm) { solverStep(B, prm); }
void afterIntegrationHost(SolverBoard& B) { afterIntegration(B); }
void deactivateBodiesHost(SolverBoard& B, const uint32_t* list, uint32_t n) { deactivateBodies(B, list, n); }

}  // namespace sv
}  // namespace eng
