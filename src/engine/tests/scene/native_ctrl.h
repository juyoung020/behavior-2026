// G4a: 로봇 제어기를 env 상태 위에서 (PhysX 없음). 재생기 --ctrl(ovd_replay.cpp ctrl_before_simulate, S2)과 같은 일:
//   에피소드 시작 뒤 simulate 마다(창 W, 전체 번호 g = W.sim): k = g - episode_start - 1, t = k / substeps.
//   t 가 바뀌면 apply_action(행동 t, 바닥·뿌리 링크 전역 자세), 서브스텝마다 step(관절 위치) -> 드라이브 목표 (위치 dof 차례, 다음 속도).
// 입력: 기록 폴더의 s1_setup.txt (capture/export_s1.py) — 행동은 부르는 쪽이 (기록 s1_actions.bin 또는 정책).
// 목표는 창의 관절체 호출(EnvArtOp type 0/1)로 바꿔 넣는다: 흐름에 적힌 로봇 드라이브 호출을 빼고 우리 것을 넣음(대조용으로 흐름 값과 비교).
#pragma once
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/omni/controllers.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_window.h"

namespace nctrl {
namespace sc2 = eng::scene;
namespace ct = eng::omni::ctrl;

struct Setup {
  uint64_t episode_start = 0;
  uint32_t substeps = 4;
  std::string base_link, root_link;
  int n_dof = 0;
  std::vector<std::string> dof_link;
  std::vector<float> sign;
  ct::R1ProConfig cfg{};
  // 묶기 (장면 파일 이름으로)
  uint32_t art = 0;                 // 로봇 관절체 번호
  std::vector<uint32_t> link;       // dof -> 링크 생성 번호
  std::vector<uint8_t> axis;        // dof -> 축
  uint32_t baseLink = 0, rootLink = 0;
  bool bound = false;

  bool load(const std::string& dir, std::string& err) {
    std::ifstream f(dir + "/s1_setup.txt");
    if (!f) {
      err = "s1_setup.txt 없음: " + dir;
      return false;
    }
    std::map<std::string, std::vector<int>> groups;
    float bl[12] = {};
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream is(line);
      std::string k;
      is >> k;
      if (k == "episode_start_post") is >> episode_start;
      else if (k == "substeps") is >> substeps;
      else if (k == "base_link") is >> base_link;
      else if (k == "root_link") is >> root_link;
      else if (k == "n_dof") {
        is >> n_dof;
        dof_link.resize(size_t(n_dof));
        sign.resize(size_t(n_dof));
      } else if (k == "group") {
        std::string g;
        int d;
        is >> g;
        while (is >> d) groups[g].push_back(d);
      } else if (k == "base_limits") {
        for (float& x : bl) {
          std::string t;
          is >> t;
          x = strtof(t.c_str(), nullptr);
        }
      } else if (k == "dof") {
        int d, sg, has;
        std::string lk, lo, hi, vlo, vhi;
        is >> d >> lk >> sg >> lo >> hi >> vlo >> vhi >> has;
        dof_link[size_t(d)] = lk;
        sign[size_t(d)] = float(sg);
        cfg.pos_lo[d] = strtof(lo.c_str(), nullptr);
        cfg.pos_hi[d] = strtof(hi.c_str(), nullptr);
        cfg.vel_lo[d] = strtof(vlo.c_str(), nullptr);
        cfg.vel_hi[d] = strtof(vhi.c_str(), nullptr);
        cfg.has_limit[d] = uint8_t(has);
      }
    }
    cfg.n_dof = n_dof;
    auto cp = [&](const char* g, int* dst, int n) {
      auto& v = groups[g];
      for (int i = 0; i < n && i < int(v.size()); ++i) dst[i] = v[size_t(i)];
    };
    cp("base", cfg.base_dof, 3);
    cp("trunk", cfg.trunk_dof, 4);
    cp("arm_left", cfg.arm_dof[0], 7);
    cp("arm_right", cfg.arm_dof[1], 7);
    cp("gripper_left", cfg.grip_dof[0], 2);
    cp("gripper_right", cfg.grip_dof[1], 2);
    for (int k = 0; k < 3; ++k) {
      cfg.base_in_lo[k] = bl[k];
      cfg.base_in_hi[k] = bl[3 + k];
      cfg.base_out_lo[k] = bl[6 + k];
      cfg.base_out_hi[k] = bl[9 + k];
    }
    ct::init(cfg);
    return true;
  }
  // dof·바닥·뿌리 링크 -> 장면 파일 관절체·링크 생성 번호·축 (첫 비잠금 축, ctrl_bind 와 같음)
  bool bind(const sc2::SceneFile& f, const sc2::EnvSolveImpl& S, std::string& err) {
    auto linkOf = [&](const std::string& name, uint32_t& a, uint32_t& l) {
      for (const sc2::SceneActor& x : f.actors)
        if (x.kind == sc2::kLink && x.name < f.names.size() && name == f.names.data() + x.name) {
          a = x.body;
          l = x.link;
          return true;
        }
      return false;
    };
    link.assign(size_t(n_dof), 0);
    axis.assign(size_t(n_dof), 0);
    for (int d = 0; d < n_dof; ++d) {
      uint32_t a = 0, l = 0;
      if (!linkOf(dof_link[size_t(d)], a, l)) {
        err = "dof 링크 없음: " + dof_link[size_t(d)];
        return false;
      }
      if (d == 0) art = a;
      if (a != art) {
        err = "dof 가 다른 관절체에 있음";
        return false;
      }
      link[size_t(d)] = l;
      const eng::art::Articulation& A = S.arts[a];
      const eng::art::JointCore& j = A.joints[eng::art::slot(A, l)];
      for (int x = 0; x < 6; ++x)
        if (j.motion[x] != 0) {
          axis[size_t(d)] = uint8_t(x);
          break;
        }
    }
    uint32_t a2 = 0;
    if (!linkOf(base_link, a2, baseLink) || a2 != art || !linkOf(root_link, a2, rootLink) || a2 != art) {
      err = "바닥/뿌리 링크 없음";
      return false;
    }
    bound = true;
    return true;
  }
};

// 판 하나의 제어기
struct Ctrl {
  ct::R1ProState st{};
  int64_t lastT = -1;
  uint64_t steps = 0, cmpN = 0, cmpBad = 0;
  long long firstBad = -1;
  Ctrl() { ct::reset(st); }

  static void linkPose(const eng::art::Articulation& A, uint32_t creation, float p[3], float q[4]) {  // getGlobalPose
    const eng::art::LinkBody& b = A.bodies[eng::art::slot(A, creation)];
    const eng::Tf g = b.body2World * eng::inverse(b.body2Actor);
    p[0] = g.p.x; p[1] = g.p.y; p[2] = g.p.z;
    q[0] = g.q.x; q[1] = g.q.y; q[2] = g.q.z; q[3] = g.q.w;
  }
  // 창 W 앞: 행동 표(acts[t][23])로 목표를 계산해 W 의 로봇 드라이브 호출을 바꾼 창을 out 에. 제어 구간 밖이면 false (W 그대로)
  bool apply(const Setup& C, const sc2::EnvSolveImpl& S, const sc2::EnvWindow& W, const float* acts, uint32_t T, sc2::EnvWindow& out) {
    if (W.sim <= C.episode_start) return false;
    const int64_t k = int64_t(W.sim - C.episode_start - 1), t = k / int64_t(C.substeps);
    if (t >= int64_t(T)) return false;
    const eng::art::Articulation& A = S.arts[C.art];
    if (t != lastT) {
      float bp[3], bq[4], rp[3], rq[4];
      linkPose(A, C.baseLink, bp, bq);
      linkPose(A, C.rootLink, rp, rq);
      ct::apply_action(C.cfg, st, acts + size_t(t) * 23, bp, bq, rp, rq);
      lastT = t;
      ++steps;
    }
    float q[ct::kMaxDof] = {};
    for (int d = 0; d < C.n_dof; ++d) {
      const eng::art::JointCore& j = A.joints[eng::art::slot(A, C.link[size_t(d)])];
      const uint32_t dofId = j.invDofIds[C.axis[size_t(d)]];
      q[d] = C.sign[size_t(d)] * A.jointPosition[A.jointData[eng::art::slot(A, C.link[size_t(d)])].jointOffset + dofId];
    }
    ct::DriveTargets o{};
    ct::step(C.cfg, st, q, o);
    // 흐름의 로봇 드라이브 호출 (대조용) 을 빼고 우리 것 (위치 dof 차례, 다음 속도)
    out = W;
    std::map<uint64_t, float> rec;
    out.artOps.clear();
    for (const sc2::EnvArtOp& op : W.artOps) {
      if (op.art == C.art && op.type <= 1) {
        rec[(uint64_t(op.type) << 40) | (uint64_t(op.link) << 8) | op.axis] = op.v;
        continue;
      }
      out.artOps.push_back(op);
    }
    for (int w = 0; w < 2; ++w)
      for (int d = 0; d < C.n_dof; ++d) {
        if (!(w == 0 ? o.set_pos[d] : o.set_vel[d])) continue;
        const float v = C.sign[size_t(d)] * (w == 0 ? o.pos[d] : o.vel[d]);
        out.artOps.push_back(sc2::EnvArtOp{C.art, uint8_t(w), C.axis[size_t(d)], {0, 0}, C.link[size_t(d)], v});
        auto it = rec.find((uint64_t(w) << 40) | (uint64_t(C.link[size_t(d)]) << 8) | C.axis[size_t(d)]);
        if (it != rec.end()) {
          ++cmpN;
          if (memcmp(&it->second, &v, 4) && !cmpBad++) firstBad = (long long)W.sim;
        }
      }
    return true;
  }
};

}  // namespace nctrl
