// omni 층 2: 판 N 개를 한꺼번에 (판 하나 = 스레드 하나 또는 블록 하나). 계산은 core/omni/*.h 의 EHD 함수 그대로 →
// 층 1(C++)과 비트 같음. 컴파일: -fmad=false -prec-div=true -prec-sqrt=true -ftz=false
//   (FTZ 끔: 제어기는 numpy, 물체 상태는 warp 라 둘 다 비정규수를 그대로 둔다. PhysX 층의 -ftz=true 와 다름에 주의)
#pragma once
#include <cstdint>

#include "core/omni/bddl.h"
#include "core/omni/controllers.h"
#include "core/omni/states.h"

namespace eng {
namespace omni {
namespace gpu {

// ---------------- BDDL: 판마다 목표 판정 (스레드 = 판) ----------------
// atom_val [boards x n_atoms], node_scratch [boards x n_nodes], head_val [boards x n_heads], success [boards]
__global__ void k_bddl_eval(const bddl::Node* nodes, int n_nodes, const int32_t* kids, const int32_t* heads, int n_heads,
                            int n_atoms, const uint8_t* atom_val, uint8_t* node_scratch, uint8_t* head_val,
                            uint8_t* success, int boards) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= boards) return;
  success[b] = bddl::eval_goal(nodes, n_nodes, kids, heads, n_heads, atom_val + (size_t)b * n_atoms,
                               node_scratch + (size_t)b * n_nodes, head_val + (size_t)b * n_heads)
                   ? 1
                   : 0;
}

// q_score: 블록 = 판, 스레드가 선택지를 나눠 맡고 최댓값을 모은다 (max 는 순서와 무관하게 정확).
__global__ void k_bddl_qscore(const int32_t* opt_start, int n_opts, const bddl::Lit* lits, int n_atoms,
                              const uint8_t* now_val, const uint8_t* init_val, const uint8_t* success, double* q_out) {
  const int b = blockIdx.x;
  const uint8_t* now = now_val + (size_t)b * n_atoms;
  const uint8_t* ini = init_val + (size_t)b * n_atoms;
  __shared__ double best[256];
  double m = 0.0;
  bool any = false;
  for (int o = threadIdx.x; o < n_opts; o += blockDim.x) {
    const int a = opt_start[o], e = opt_start[o + 1];
    double s = 0.0;
    if (e > a) {
      long long newly = 0;
      for (int k = a; k < e; ++k) newly += (!bddl::lit_val(lits[k], ini)) && bddl::lit_val(lits[k], now);
      s = (double)newly / (double)(e - a);
    }
    if (!any || s > m) m = s;
    any = true;
  }
  best[threadIdx.x] = any ? m : -1.0;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st && best[threadIdx.x + st] > best[threadIdx.x]) best[threadIdx.x] = best[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double q;
    if (success[b]) q = 1.0;
    else if (n_opts == 0) q = 0.0;
    else q = best[0];
    q_out[b] = q;
  }
}

// ---------------- R1Pro 제어기 (스레드 = 판) ----------------
// 행동 [boards x 23], 바닥/뿌리 자세, 관절 위치 [boards x n_dof] → 드라이브 목표
__global__ void k_ctrl_step(const ctrl::R1ProConfig* cfg, ctrl::R1ProState* state, const uint8_t* has_action,
                            const float* action, const float* base_p, const float* base_q, const float* root_p,
                            const float* root_q, const float* joint_pos, ctrl::DriveTargets* out, int boards) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= boards) return;
  const ctrl::R1ProConfig c = *cfg;
  ctrl::R1ProState s = state[b];
  if (has_action[b])
    ctrl::apply_action(c, s, action + (size_t)b * 23, base_p + (size_t)b * 3, base_q + (size_t)b * 4,
                       root_p + (size_t)b * 3, root_q + (size_t)b * 4);
  ctrl::step(c, s, joint_pos + (size_t)b * c.n_dof, out[b]);
  state[b] = s;
}

// ---------------- 물체 상태 ----------------
__global__ void k_pose_to_mat(const float* poses, wf::M44* mats, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mats[i] = wf::pose_to_mat(poses + (size_t)i * 7);
}

// 판 하나 안에서: 링크 l, 방향 k 의 광선을 물체 a 의 AABB 중심에서 쏜다 (스레드 = (s, a, l, k))
__global__ void k_adjacency(const float* aabb, int O, const int32_t* aidx, int N, const wf::M44* mats, const uint8_t* has_mesh,
                            const int32_t* l2o, const int32_t* l2s, int L, const float* dirs, const float* maxd, int K,
                            const float* pts, const int32_t* tri, const int32_t* poff, const int32_t* toff, int S,
                            uint8_t* adj) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)S * N * L * K;
  if (tid >= total) return;
  const int k = (int)(tid % K);
  const int l = (int)((tid / K) % L);
  const int a = (int)((tid / ((long long)K * L)) % N);
  const int s = (int)(tid / ((long long)K * L * N));
  const int ai = aidx[a];
  if (ai < 0 || !has_mesh[l] || l2s[l] != s) return;
  const int b = l2o[l];
  if (b < 0 || b == a) return;
  float o_w[3], o_l[3], d_l[3];
  st::aabb_center(aabb + ((size_t)s * O + ai) * 6, o_w);
  st::adjacency_ray_local(mats[l], o_w, dirs + k * 3, o_l, d_l);
  if (st::mesh_anyhit_bruteforce(pts + (size_t)poff[l] * 3, tri + (size_t)toff[l] * 3, toff[l + 1] - toff[l], o_l, d_l,
                                 maxd[k]))
    adj[(((size_t)s * N + a) * N + b) * K + k] = 1;  // OR (여러 스레드가 1 만 씀)
}

}  // namespace gpu
}  // namespace omni
}  // namespace eng
