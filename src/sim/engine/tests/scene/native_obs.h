// G4c: 관측 proprio 61 을 env 상태에서 (core/omni/obs.h, 공식 _get_proprioception_dict 와 비트 같음 — tests/omni/test_obs.py).
// 입력: 기록 폴더 obs_setup.txt (export_obs_setup.py). 관절 위치·속도 = 텐서 API dof 차례(제어기 Setup 의 링크·축·부호), 링크 자세 = getGlobalPose.
// 바닥 회전 cos/sin 은 MKL VML (ENGINE_MKL_LIB = libtorch_cpu.so 경로가 있어야 함, core/omni/mkl_trig.h).
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/omni/obs.h"
#include "tests/scene/native_ctrl.h"

namespace nobs {
namespace ob = eng::omni::obs;

struct Setup {
  std::string base, eef[2];
  ob::R1ProIdx ix{};
  uint32_t baseLink = 0, eefLink[2] = {0, 0};
  bool bound = false;
  bool load(const std::string& dir, std::string& err) {
    std::ifstream f(dir + "/obs_setup.txt");
    if (!f) {
      err = "obs_setup.txt 없음 (export_obs_setup.py)";
      return false;
    }
    std::vector<int> idx;
    for (std::string line; std::getline(f, line);) {
      std::istringstream is(line);
      std::string k;
      is >> k;
      if (k == "base_link") is >> base;
      else if (k == "eef_left") is >> eef[0];
      else if (k == "eef_right") is >> eef[1];
      else if (k == "idx")
        for (int d; is >> d;) idx.push_back(d);
    }
    if (idx.size() != 26) {
      err = "obs_setup idx 가 26 개가 아님";
      return false;
    }
    int k = 0;
    for (int i = 0; i < 3; ++i) ix.base[i] = idx[size_t(k++)];
    ix.yaw_dof = idx[size_t(k++)];
    for (int i = 0; i < 4; ++i) ix.trunk[i] = idx[size_t(k++)];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 7; ++i) ix.arm[a][i] = idx[size_t(k++)];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 2; ++i) ix.grip[a][i] = idx[size_t(k++)];
    return true;
  }
  bool bind(const eng::scene::SceneFile& f, const nctrl::Setup& C, std::string& err) {
    auto linkOf = [&](const std::string& name, uint32_t& l) {
      for (const eng::scene::SceneActor& x : f.actors)
        if (x.kind == eng::scene::kLink && x.body == C.art && x.name < f.names.size() && name == f.names.data() + x.name) {
          l = x.link;
          return true;
        }
      return false;
    };
    if (!linkOf(base, baseLink) || !linkOf(eef[0], eefLink[0]) || !linkOf(eef[1], eefLink[1])) {
      err = "관측 링크를 못 찾음";
      return false;
    }
    bound = true;
    return true;
  }
  static ob::Pose pose(const eng::art::Articulation& A, uint32_t creation) {
    ob::Pose o;
    nctrl::Ctrl::linkPose(A, creation, o.p, o.q);
    return o;
  }
  // proprio 61 (판 하나)
  void proprio(const nctrl::Setup& C, const eng::scene::EnvSolveImpl& S, float out[61]) const {
    const eng::art::Articulation& A = S.arts[C.art];
    float jp[eng::omni::ctrl::kMaxDof] = {}, jv[eng::omni::ctrl::kMaxDof] = {};
    for (int d = 0; d < C.n_dof; ++d) {
      const uint32_t sl = eng::art::slot(A, C.link[size_t(d)]);
      const uint32_t i = A.jointData[sl].jointOffset + A.joints[sl].invDofIds[C.axis[size_t(d)]];
      jp[d] = C.sign[size_t(d)] * A.jointPosition[i];
      jv[d] = C.sign[size_t(d)] * A.jointVelocity[i];
    }
    const ob::Pose b = pose(A, baseLink);
    const ob::Pose e[2] = {pose(A, eefLink[0]), pose(A, eefLink[1])};
    ob::proprio_r1pro(ix, jp, jv, b, e, out);
  }
};

}  // namespace nobs
