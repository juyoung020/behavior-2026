// omni 시험 5: 보조 잡기 판단 — 우리 C++ (core/omni/assisted_grasp.h) = 공식 robot.py 메서드 (gen_ag_ref.py 정답).
//   test_ag <~/engine-data/omni/ag>
// 매 서브스텝, 팔마다: 사건(없음/놓기/잡기 시도)·잡을 링크·팔 상태(쥔 물체, 놓기 수, 잡기 수)를 비교. 관절 종류 판정도 비교.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/omni/assisted_grasp.h"

using namespace eng::omni;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_ag <dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  FILE* f = fopen((dir + "/ag.bin").c_str(), "rb");
  if (!f) return 2;
  int steps, nl, robot_link;
  fread(&steps, 4, 1, f), fread(&nl, 4, 1, f), fread(&robot_link, 4, 1, f);
  std::vector<int32_t> link_obj(nl);
  std::vector<uint8_t> link_dyn(nl);
  fread(link_obj.data(), 4, nl, f), fread(link_dyn.data(), 1, nl, f);
  ag::Params p;
  ag::ArmState st[2];
  long long cmp = 0, bad = 0, n_try = 0, n_rel = 0, n_grasp = 0;
  std::string first;
  for (int s = 0; s < steps; ++s) {
    for (int arm = 0; arm < 2; ++arm) {
      uint8_t applying, success;
      float eef[3];
      std::vector<uint8_t> cm(nl), rm(nl), nf(nl);
      std::vector<float> pos(nl * 3);
      fread(&applying, 1, 1, f), fread(eef, 4, 3, f), fread(cm.data(), 1, nl, f), fread(rm.data(), 1, nl, f);
      fread(nf.data(), 1, nl, f), fread(pos.data(), 4, nl * 3, f), fread(&success, 1, 1, f);
      uint8_t et;
      int32_t tl, inh, rc, gc;
      fread(&et, 1, 1, f), fread(&tl, 4, 1, f), fread(&inh, 4, 1, f), fread(&rc, 4, 1, f), fread(&gc, 4, 1, f);
      // 후보 목록 (_find_gripper_contacts: 로봇 자기 링크 제외)
      ag::CandidateIn c{};
      c.n = 0;
      for (int i = 0; i < nl; ++i) {
        if (!cm[i] || i == robot_link) continue;
        const int k = c.n++;
        c.link[k] = i;
        c.fingers[k] = nf[i];
        c.ray_hit[k] = rm[i];
        c.obj[k] = link_obj[i];
        c.dynamic[k] = link_dyn[i];
        memcpy(c.pos[k], &pos[i * 3], 12);
      }
      int in_hand = -1;
      if (st[arm].obj_in_hand < 0 && applying) in_hand = ag::calculate_in_hand(c, eef);
      int target = -1;
      const ag::Event ev = ag::step_arm(p, st[arm], applying != 0, in_hand, &target);
      const int tlink = target >= 0 ? c.link[target] : -1;
      if (ev == ag::EV_TRY_GRASP) {
        ++n_try;
        if (success) {
          ag::grasp_established(st[arm], c.obj[target], tlink);
          ++n_grasp;
        }
      }
      if (ev == ag::EV_RELEASE) ++n_rel;
      const long long before = bad;
      cmp += 5;
      bad += (int)ev != (int)et;
      bad += tlink != tl;
      bad += st[arm].obj_in_hand != inh;
      bad += st[arm].release_counter != rc;
      bad += st[arm].grasp_counter != gc;
      if (bad != before && first.empty()) {
        char b[256];
        snprintf(b, sizeof b, "step %d arm %d: 사건 %d/%d 링크 %d/%d 쥔 %d/%d 놓기 %d/%d 잡기 %d/%d", s, arm, ev, et, tlink, tl,
                 st[arm].obj_in_hand, inh, st[arm].release_counter, rc, st[arm].grasp_counter, gc);
        first = b;
        // 이후 비교가 의미 있게 정답 상태로 맞춘다
      }
      st[arm].obj_in_hand = inh;
      st[arm].release_counter = rc;
      st[arm].grasp_counter = gc;
    }
  }
  fclose(f);
  printf("보조 잡기 판단   서브스텝 %d x 팔 2, 비교 %lld, 다름 %lld (잡기 시도 %lld, 성공 %lld, 놓기 %lld) %s\n", steps, cmp, bad, n_try,
         n_grasp, n_rel, first.c_str());
  // 관절 종류
  f = fopen((dir + "/jt.bin").c_str(), "rb");
  int n;
  fread(&n, 4, 1, f);
  long long jb = 0;
  for (int i = 0; i < n; ++i) {
    float mass;
    uint8_t fixed, root, anc, code;
    fread(&mass, 4, 1, f), fread(&fixed, 1, 1, f), fread(&root, 1, 1, f), fread(&anc, 1, 1, f), fread(&code, 1, 1, f);
    jb += ag::joint_type(p, mass, fixed != 0, root != 0, anc != 0) != code;
  }
  fclose(f);
  printf("보조 잡기 관절 종류   비교 %d, 다름 %lld\n", n, jb);
  return (bad || jb) ? 1 : 0;
}
