// 빌드 확인: core/contact/scene_step.h (contact 장면 한 서브스텝) 가 solver 경계 자료형과 함께 컴파일되는지. 실행하면 빈 장면 한 스텝.
#include "core/contact/scene_step.h"

#include <cstdio>

int main() {
  eng::contact::ContactScene S;
  eng::contact::SolverInputOut out;
  std::vector<eng::sv::SolverCM> cms(4);
  eng::contact::contactSolverInput(S, nullptr, nullptr, nullptr, cms.data(), uint32_t(cms.size()), out);
  printf("scene_step 빌드 확인: 패치 %zu 점 %zu\n", out.patches.size(), out.contacts.size());
  return 0;
}
