// omni 층 2: 판 N 개를 한꺼번에 (판 하나 = 스레드 하나 또는 블록 하나). 계산은 core/omni/*.h 의 EHD 함수 그대로 →
// 층 1(C++)과 비트 같음. 컴파일: -fmad=false -prec-div=true -prec-sqrt=true -ftz=false
//   (FTZ 끔: 제어기는 numpy, 물체 상태는 warp 라 둘 다 비정규수를 그대로 둔다. PhysX 층의 -ftz=true 와 다름에 주의)
#pragma once
#include <cstdint>

#include "core/omni/agframe.h"
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

// AABB: 스레드 = 물체 행 (링크·꼭짓점을 차례로, min/max 는 순서와 무관 — 부호 있는 0 동률만 예외)
__global__ void k_aabb(const wf::M44* mats, const float* poses, const int32_t* link_row, const uint8_t* has_mesh,
                       const float* pts, const int32_t* poff, int L, const int32_t* base_link, int n_base, int n_rows,
                       float* aabb) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) return;
  float* a = aabb + (size_t)row * 6;
  st::aabb_init(a);
  for (int l = 0; l < L; ++l) {
    if (link_row[l] != row || !has_mesh[l]) continue;
    for (int v = poff[l]; v < poff[l + 1]; ++v) {
      float w[3];
      st::aabb_vertex_world(mats[l], pts + (size_t)v * 3, w);
      st::aabb_accumulate(a, w);
    }
  }
  for (int i = 0; i < n_base; ++i)
    if (link_row[base_link[i]] == row) st::aabb_fallback(a, poses + (size_t)base_link[i] * 7);
}

// Inside: 스레드 = (s, i, j) 판정 한 칸. 메시·면을 차례로 (공식은 면마다 스레드 + 원자 OR — 결과 같음)
__global__ void k_inside(const float* aabb, int O, const int32_t* aidx, int N, int S, const wf::M44* mats, int M,
                         const int32_t* mcont, const int32_t* mscene, const int32_t* mpar, const wf::M44* minv,
                         const int32_t* fstart, const float* fc, const float* fn, uint8_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= S * N * N) return;
  const int j = t % N, i = (t / N) % N, s = t / (N * N);
  const float* aabb_s = aabb + (size_t)s * O * 6;
  uint8_t v = 0;
  if (i != j && st::inside_prefilter(aabb_s, aidx, i, j)) {
    float c[3];
    st::aabb_center(aabb_s + aidx[i] * 6, c);
    for (int m = 0; m < M && !v; ++m) {
      if (mcont[m] != j || mscene[m] != s) continue;
      const wf::M44 iw = st::inside_inv_world(mats[mpar[m]], minv[m]);
      bool outside = false;
      for (int f = fstart[m]; f < fstart[m + 1] && !outside; ++f) outside = st::inside_face_outside(iw, c, fc + f * 3, fn + f * 3);
      if (!outside) v = 1;
    }
  }
  out[t] = v;
}

// ToggledOn 표식 겹침: 스레드 = (표식, 손가락) 쌍. mask 가 1 인 물체만, 겹치면 2 (원자 max 대신 1→2 쓰기: 같은 값만 씀)
__global__ void k_toggle_overlap(const wf::M44* mats, const uint8_t* has_mesh, const float* pts, const int32_t* tri,
                                 const int32_t* poff, const int32_t* toff, const int32_t* mk_parent, const float* mk_off, const float* mk_rad,
                                 const int32_t* pairs, int n_pairs, int32_t* mask) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_pairs) return;
  const int k = pairs[p * 2], fg = pairs[p * 2 + 1];
  if (mask[k] != 1 || !has_mesh[fg]) return;  // (공식도 같은 경쟁: 1 을 본 스레드만 2 를 쓴다)
  if (st::toggle_marker_overlap(mats[mk_parent[k]], mk_off + k * 3, mk_rad[k], mats[fg], pts + (size_t)poff[fg] * 3,
                                tri + (size_t)toff[fg] * 3,
                                toff[fg + 1] - toff[fg]))
    atomicMax(&mask[k], 2);
}

// 보조 잡기 관절 틀: 스레드 = 한 번의 틀 계산 (행 = contact3 link_pos3 link_q4 scale3 → pos3 q4)
__global__ void k_agframe(const float* in, int n, float* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float* x = in + (size_t)t * 13;
  agf::grasp_frame(x, x + 3, x + 6, x + 10, out + (size_t)t * 7, out + (size_t)t * 7 + 3);
}

// 온도 사슬 한 스텝 (층 1 과 같은 함수, 원자 덧셈 대신 스레드 = (s, n) 가 열원 번호 순으로 더함 — 결정적)
struct HeatGates {
  const uint8_t *req_tg, *req_cl, *req_fi;
  const int32_t *tg_idx, *op_idx, *fi_idx;
  const uint8_t* tg_v;
  int n_tg, n_tg_sc;
  const uint8_t* op_v;
  int n_op, n_op_sc;
  const uint8_t* fi_v;  // 앞 스텝 OnFire
  int n_fi, n_fi_sc;
};
__global__ void k_heat_active(HeatGates g, int S, int H, uint8_t* hv) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= S * H) return;
  const int s = t / H, h = t % H;
  hv[t] = st::heat_active(g.req_tg, g.req_cl, g.req_fi, g.tg_idx, g.op_idx, g.fi_idx, g.tg_v, g.n_tg, g.n_tg_sc, g.op_v, g.n_op,
                          g.n_op_sc, g.fi_v, g.n_fi, g.n_fi_sc, s, h);
}
// in.temp_vals 는 이번 스텝 들어갈 때 온도. temp_out 에 감쇠·불 유지 뒤 온도, infl (S,H,N), inc (S,N) 는 확인용
__global__ void k_heat_gather(st::HeatIn in, int S, const uint8_t* req_fi, const float* ign, float def, float decay,
                              float dt, uint8_t* infl, float* inc, float* temp_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = in.n_temp, H = in.N_hss;
  if (t >= S * N) return;
  const int s = t / N, n = t % N;
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    float dl = 0.0f;
    const bool hit = st::heat_pair(in, s, h, n, &dl);
    infl[((size_t)s * H + h) * N + n] = hit ? 1 : 0;
    if (hit) acc = acc + dl;
  }
  inc[t] = acc;
  float v = st::temp_decay(in.temp_vals[t], acc, def, decay, dt);
  for (int h = 0; h < H; ++h)
    if (in.self_temp[h] == n) v = st::temp_self_clamp(req_fi[h], in.src_vals[s * H + h], in.src_temp[h], ign[h], v);
  temp_out[t] = v;
}
__global__ void k_max_fire(const float* temp, int S, int N, float* mx, const int32_t* mx_idx, const int32_t* of_idx,
                           const float* of_ign, uint8_t* fire) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= S * N) return;
  const int s = t / N, o = t % N;
  mx[t] = st::max_temp_update(mx[t], temp[s * N + mx_idx[o]]);
  const int ti = of_idx[o];
  fire[t] = st::on_fire_value(ti, ti < 0 ? 0.0f : temp[s * N + ti], of_ign[o]);
}

}  // namespace gpu
}  // namespace omni
}  // namespace eng
