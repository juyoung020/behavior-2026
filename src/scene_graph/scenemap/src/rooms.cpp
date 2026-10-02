// 방 나누기 구현(include/scenemap/rooms.hpp).
#include "scenemap/rooms.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>

namespace scenemap {
namespace {

constexpr float kInf = 1e20f;

// Felzenszwalb–Huttenlocher 1D 제곱 거리 변환
void dt1d(const float* f, int n, float* d, int* v, float* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  auto isect = [&](int q, int r) { return ((f[q] + float(q) * q) - (f[r] + float(r) * r)) / (2.f * (q - r)); };
  for (int q = 1; q < n; ++q) {
    float s = isect(q, v[k]);
    while (s <= z[k] && k > 0) {
      --k;
      s = isect(q, v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const float dq = float(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

// free 칸의 막힌 칸(그리고 격자 밖)까지 제곱 거리(칸²)
std::vector<float> edt(const std::vector<uint8_t>& free, int W, int H) {
  // 세로(1D, 이진이라 두 번 쓸기): 위·아래로 가장 가까운 막힌 칸(격자 밖 = 막힘)까지 칸 수, 행 단위로 쓸어 캐시 친화
  const int PW = W + 2;
  std::vector<float> g(size_t(H) * PW, 0.f);
  std::vector<int> run(W, 0);
  for (int y = 0; y < H; ++y) {   // 아래(−y) 쪽 거리
    const uint8_t* fr = &free[size_t(y) * W];
    float* row = &g[size_t(y) * PW + 1];
    for (int x = 0; x < W; ++x) {
      run[x] = fr[x] ? run[x] + 1 : 0;
      row[x] = float(run[x]);
    }
  }
  std::fill(run.begin(), run.end(), 0);
  for (int y = H - 1; y >= 0; --y) {   // 위(+y) 쪽과 작은 것, 제곱
    const uint8_t* fr = &free[size_t(y) * W];
    float* row = &g[size_t(y) * PW + 1];
    for (int x = 0; x < W; ++x) {
      run[x] = fr[x] ? run[x] + 1 : 0;
      const float m = std::min(row[x], float(run[x]));
      row[x] = m * m;
    }
  }
  // 가로: 아래 봉투(Felzenszwalb–Huttenlocher), 양 끝 덧댐 칸 = 막힘
  std::vector<float> d(PW), z(PW + 1);
  std::vector<int> v(PW);
  std::vector<float> out(size_t(W) * H);
  for (int y = 0; y < H; ++y) {
    float* row = &g[size_t(y) * PW];
    dt1d(row, PW, d.data(), v.data(), z.data());
    std::memcpy(&out[size_t(y) * W], d.data() + 1, sizeof(float) * W);
  }
  return out;
}

struct DSU {
  std::vector<int> p;
  int find(int a) {
    while (p[a] != a) a = p[a] = p[p[a]];
    return a;
  }
};

// 작은 모름 구멍 → 빈칸, 빈칸에 둘러싸인 작은 점유 점 → 빈칸
void cleanFree(const std::vector<uint8_t>& cls, int W, int H, int hole_cells, int speck_cells, std::vector<uint8_t>& free) {
  const size_t N = size_t(W) * H;
  std::vector<uint8_t> seen(N, 0);
  std::vector<int> stack, comp;
  for (size_t s = 0; s < N; ++s) {
    if (cls[s] == 1 || seen[s]) continue;
    const uint8_t want = cls[s];   // 0 모름, 2 점유
    const bool eight = want == 2;
    const int limit = want == 0 ? hole_cells : speck_cells;
    stack.assign(1, int(s));
    seen[s] = 1;
    comp.clear();
    bool border = false, keep = limit > 0;
    int nfree = 0, nother = 0;
    while (!stack.empty()) {
      const int c = stack.back();
      stack.pop_back();
      if (keep) comp.push_back(c);
      if (int(comp.size()) > limit) { keep = false; comp.clear(); }
      const int x = c % W, y = c / W;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          const bool diag = dx && dy;
          const int X = x + dx, Y = y + dy;
          if (X < 0 || Y < 0 || X >= W || Y >= H) { border = true; continue; }
          const size_t j = size_t(Y) * W + X;
          if (cls[j] == want) {
            if ((!diag || eight) && !seen[j]) { seen[j] = 1; stack.push_back(int(j)); }
          } else if (!diag) {
            if (cls[j] == 1) ++nfree; else ++nother;
          }
        }
    }
    if (!keep || border || comp.empty()) continue;
    const bool ok = want == 0 ? (nfree > 0 && 2 * nfree >= nfree + nother) : nother == 0;
    if (ok)
      for (int c : comp) free[size_t(c)] = 1;
  }
}

}  // namespace

uint32_t RoomSeg::at(double x, double y) const {
  if (ids.empty()) return 0;
  const int ix = int(std::floor(x / res)) - gx0, iy = int(std::floor(y / res)) - gy0;
  if (ix < 0 || iy < 0 || ix >= w || iy >= h) return 0;
  return ids[size_t(iy) * w + ix];
}

int RoomSeg::index(uint32_t id) const {
  for (size_t i = 0; i < rooms.size(); ++i)
    if (rooms[i].id == id) return int(i);
  return -1;
}

std::shared_ptr<RoomSeg> segmentRooms(const GridView& g, const RoomParams& P) {
  const auto t0 = std::chrono::steady_clock::now();
  auto S = std::make_shared<RoomSeg>();
  S->w = g.w; S->h = g.h; S->gx0 = g.gx0; S->gy0 = g.gy0; S->res = g.res;
  const int W = g.w, H = g.h;
  const size_t N = size_t(W) * H;
  if (!N || !g.cells) return S;
  const double res = g.res, a1 = res * res;
  // 1) 분류·정리
  std::vector<uint8_t> cls(N);
  S->rawfree.resize(N);
  for (size_t i = 0; i < N; ++i) {
    const int v = g.cells[i];
    cls[i] = (v >= 0 && v <= P.free_max) ? 1 : (v >= P.occ_min ? 2 : 0);
    S->rawfree[i] = cls[i] == 1;
  }
  std::vector<uint8_t> free = S->rawfree;
  cleanFree(cls, W, H, int(P.hole_m2 / a1), int(P.speck_m2 / a1 + 1e-9), free);
  // 테두리 빈칸은 잠시 막음(이웃 범위 검사 없이) — 끝에서 안쪽 이웃의 방을 받음
  std::vector<int> rim;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if ((x == 0 || y == 0 || x == W - 1 || y == H - 1) && free[size_t(y) * W + x]) {
        rim.push_back(y * W + x);
        free[size_t(y) * W + x] = 0;
      }
  // 2) 거리 변환 → 여유(벽 면까지, m)
  const std::vector<float> d2 = edt(free, W, H);
  std::vector<float> clear(N, 0.f);
  for (size_t i = 0; i < N; ++i)
    if (free[i]) clear[i] = float(std::sqrt(d2[i]) * res - res / 2);
  // 3) 거름: 높은 문턱부터 칸을 넣고 union-find, 수명 긴 성분을 씨앗으로 얼림
  const int K = std::clamp(int(std::floor((P.dil_max - P.dil_min) / P.dil_step + 1e-9)) + 1, 1, 120);
  const int life = std::max(1, int(std::lround(P.min_life / P.dil_step)));
  const int min_seed = std::max(1, int(P.min_seed_m2 / a1));
  // 칸을 문턱 층으로(세기 정렬): order[start[L] .. start[L+1])
  std::vector<int8_t> lv(N, -1);
  const float inv_step = float(1.0 / P.dil_step);
  std::vector<int> start(K + 1, 0);
  for (size_t i = 0; i < N; ++i) {
    if (!free[i] || clear[i] <= P.dil_min) continue;
    const float x = (clear[i] - float(P.dil_min)) * inv_step;
    const int L = std::clamp(int(std::ceil(x - 1e-5f)) - 1, 0, K - 1);
    lv[i] = int8_t(L);
    start[L + 1]++;
  }
  for (int L = 0; L < K; ++L) start[L + 1] += start[L];
  std::vector<int> order(start[K]);
  {
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t i = 0; i < N; ++i)
      if (lv[i] >= 0) order[fill[lv[i]]++] = int(i);
  }
  struct Span {
    const int* b; const int* e;
    const int* begin() const { return b; }
    const int* end() const { return e; }
  };
  auto bucket = [&](int L) { return Span{order.data() + start[L], order.data() + start[L + 1]}; };
  std::vector<int> par(N, -1), size(N, 0), birth(N, 0), csid(N, -1);   // csid: 뿌리의 씨앗 id(−1 = 여럿)
  std::vector<int> cell_sid(N, -1);
  DSU sid;
  std::vector<uint8_t> sdead;
  auto findc = [&](int a) {
    while (par[a] != a) a = par[a] = par[par[a]];
    return a;
  };
  int nbig = 0;
  auto sig = [&](int r, int L) { return size[r] >= min_seed && birth[r] - L >= life; };
  auto kill = [&](int s) { sdead[sid.find(s)] = 1; };
  for (int L = K - 1; L >= 0; --L) {
    for (int c : bucket(L)) {
      const int nb[4] = {c - 1, c + 1, c - W, c + W};   // 빈칸은 테두리에 없음(아래 rim)
      int first = -1;
      for (int j : nb)
        if (j >= 0 && par[j] >= 0) { first = j; break; }
      if (first < 0) {   // 새 성분, 새 씨앗
        par[c] = c;
        size[c] = 1;
        birth[c] = L;
        const int s = int(sid.p.size());
        sid.p.push_back(s);
        sdead.push_back(0);
        csid[c] = s;
        cell_sid[c] = s;
        if (min_seed <= 1) ++nbig;
      } else {           // 이웃 성분에 붙음(혼자 칸은 언제나 짧게 산 쪽: 홀로 성분이면 그 씨앗, 섞인 성분이면 넘치기가 정함)
        const int r = findc(first);
        par[c] = r;
        nbig += size[r] + 1 == min_seed;
        size[r]++;
        cell_sid[c] = csid[r];
      }
      for (int j : nb) {
        if (j < 0 || j == first || par[j] < 0) continue;
        int ra = findc(c), rb = findc(j);
        if (ra == rb) continue;
        const bool bigA = size[ra] >= min_seed, bigB = size[rb] >= min_seed;
        // ra = 나이 많은 쪽(태어난 문턱 높음, 같으면 큰 쪽)
        if (birth[rb] > birth[ra] || (birth[rb] == birth[ra] && size[rb] > size[ra])) std::swap(ra, rb);
        const int sa = csid[ra], sb = csid[rb];
        int out_sid;
        if (sa < 0 && sb < 0) {
          out_sid = -1;
        } else if (sa < 0 || sb < 0) {
          const int single = sa < 0 ? rb : ra;
          if (!sig(single, L)) kill(csid[single]);
          out_sid = -1;
        } else if (sig(ra, L) && sig(rb, L)) {
          out_sid = -1;
        } else {
          // 짧게 산 쪽을 흡수(둘 다 짧으면 어린 쪽을 나이 많은 쪽에)
          const bool keepA = sig(ra, L) || !sig(rb, L);
          const int keep = keepA ? sa : sb, lose = keepA ? sb : sa;
          sid.p[sid.find(lose)] = sid.find(keep);
          out_sid = sid.find(keep);
        }
        par[rb] = ra;
        size[ra] += size[rb];
        csid[ra] = out_sid;
        nbig += (size[ra] >= min_seed) - bigA - bigB;
      }
      // 섞인 성분(여럿)에 바로 든 칸의 자기 씨앗은 위 합치기에서 죽음 → 넘치기가 정함
    }
    S->filtration.push_back({P.dil_min + L * P.dil_step, nbig});
  }
  // 남은 홀로 성분: 크면 씨앗
  for (int L = 0; L < K; ++L)
    for (int c : bucket(L))
      if (par[c] == c && csid[c] >= 0 && size[c] < min_seed) kill(csid[c]);
  std::vector<int> lab(N, 0);
  std::vector<int> seed_of(sid.p.size(), 0);   // 씨앗 뿌리 → 1..n
  int nseed = 0;
  for (int L = 0; L < K; ++L)
    for (int c : bucket(L)) {
      if (cell_sid[c] < 0) continue;
      const int r = sid.find(cell_sid[c]);
      if (sdead[r]) continue;
      if (!seed_of[r]) seed_of[r] = ++nseed;
      lab[c] = seed_of[r];
    }
  S->n_seeds = nseed;
  // 4) 넘치기: 여유 큰 칸부터(반 칸 단위 통), 같은 통은 먼저 온 순
  int nreg = S->n_seeds;
  {
    const float q = float(2.0 / res);
    float cmax = 0;
    for (size_t i = 0; i < N; ++i) cmax = std::max(cmax, clear[i]);
    const int NB = int(cmax * q) + 2;
    std::vector<std::vector<int>> qb(NB);
    auto key = [&](size_t i) { return std::clamp(int(std::max(0.f, clear[i]) * q), 0, NB - 1); };
    for (int y = 0; y < H; ++y)   // 처음 앞줄: 이웃에 아직 정해지지 않은 빈칸이 있는 씨앗 칸만
      for (int x = 0; x < W; ++x) {
        const size_t i = size_t(y) * W + x;
        if (!lab[i]) continue;
        const bool fr = (x > 0 && free[i - 1] && !lab[i - 1]) || (x + 1 < W && free[i + 1] && !lab[i + 1]) ||
                        (y > 0 && free[i - W] && !lab[i - W]) || (y + 1 < H && free[i + W] && !lab[i + W]);
        if (fr) qb[key(i)].push_back(int(i));
      }
    for (int k = NB - 1; k >= 0; --k) {
      std::vector<int>& b = qb[k];
      for (size_t h = 0; h < b.size(); ++h) {
        const int c = b[h];
        const int nb[4] = {c - 1, c + 1, c - W, c + W};   // 빈칸은 테두리에 없음(아래 rim)
        for (int j : nb) {
          if (j < 0 || !free[j] || lab[j]) continue;
          lab[j] = lab[c];
          const int kj = std::min(k, key(size_t(j)));
          qb[kj].push_back(j);
        }
      }
      std::vector<int>().swap(b);
    }
    // 씨앗 없는 빈칸 덩이
    const int min_room = int(P.min_room_m2 / a1);
    std::vector<int> st, comp;
    for (size_t s = 0; s < N; ++s) {
      if (!free[s] || lab[s]) continue;
      comp.clear();
      st.assign(1, int(s));
      lab[s] = -1;
      while (!st.empty()) {
        const int c = st.back();
        st.pop_back();
        comp.push_back(c);
        const int nb[4] = {c - 1, c + 1, c - W, c + W};   // 빈칸은 테두리에 없음(아래 rim)
        for (int j : nb)
          if (j >= 0 && free[j] && !lab[j]) { lab[j] = -1; st.push_back(j); }
      }
      const int v = int(comp.size()) >= min_room ? ++nreg : -1;
      for (int c : comp) lab[c] = v;
    }
    for (int& v : lab) v = std::max(v, 0);
  }
  // 5) 합치기(긴 이음매 · 작은 방)
  struct Seam { int n = 0; double sx = 0, sy = 0; float cmax = 0; };
  auto seams = [&](const std::vector<int>& L, std::unordered_map<uint64_t, Seam>& out) {
    out.clear();
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const size_t c = size_t(y) * W + x;
        const int a = L[c];
        if (!a) continue;
        for (int k = 0; k < 2; ++k) {
          const int X = x + (k == 0), Y = y + (k == 1);
          if (X >= W || Y >= H) continue;
          const size_t j = size_t(Y) * W + X;
          const int b = L[j];
          if (!b || b == a) continue;
          const uint64_t key = (uint64_t(std::min(a, b)) << 32) | uint32_t(std::max(a, b));
          Seam& s = out[key];
          ++s.n;
          s.sx += (x + X) * 0.5;
          s.sy += (y + Y) * 0.5;
          s.cmax = std::max(s.cmax, std::min(clear[c], clear[j]));
        }
      }
  };
  std::unordered_map<uint64_t, Seam> sm;
  seams(lab, sm);
  DSU rg;
  rg.p.resize(nreg + 1);
  std::iota(rg.p.begin(), rg.p.end(), 0);
  std::vector<int> area(nreg + 1, 0);
  for (int v : lab) area[v]++;
  const int max_door = P.max_door_m > 0 ? int(std::lround(P.max_door_m / res)) : 1 << 30;
  for (auto& [k, s] : sm)
    if (s.n >= max_door) {
      const int a = rg.find(int(k >> 32)), b = rg.find(int(k & 0xffffffffu));
      if (a != b) { rg.p[b] = a; area[a] += area[b]; }
    }
  const int min_room = int(P.min_room_m2 / a1);
  std::vector<uint8_t> dropped(nreg + 1, 0);
  while (true) {   // 가장 작은 방부터 이음매가 가장 긴 이웃에
    int small = -1;
    for (int r = 1; r <= nreg; ++r)
      if (rg.find(r) == r && !dropped[r] && area[r] < min_room && (small < 0 || area[r] < area[small])) small = r;
    if (small < 0) break;
    std::unordered_map<int, int> nbn;
    for (auto& [k, s] : sm) {
      const int a = rg.find(int(k >> 32)), b = rg.find(int(k & 0xffffffffu));
      if (a == b) continue;
      if (a == small) nbn[b] += s.n;
      else if (b == small) nbn[a] += s.n;
    }
    int best = -1, bn = 0;
    for (auto& [r, n] : nbn)
      if (n > bn || (n == bn && r < best)) { bn = n; best = r; }
    if (best < 0) { dropped[small] = 1; continue; }
    rg.p[small] = best;
    area[best] += area[small];
  }
  // 다시 번호(1..n)
  std::vector<int> remap(nreg + 1, 0);
  int n = 0;
  for (int r = 1; r <= nreg; ++r) {
    const int root = rg.find(r);
    if (dropped[root]) continue;
    if (!remap[root]) remap[root] = ++n;
    remap[r] = remap[root];
  }
  for (int& v : lab) v = v ? remap[v] : 0;
  if (W >= 3 && H >= 3)
    for (int c : rim) {
      const int x = std::clamp(c % W, 1, W - 2), y = std::clamp(c / W, 1, H - 2);
      lab[c] = lab[size_t(y) * W + x];
    }
  // 6) 방 모양·문
  S->rooms.resize(n);
  std::vector<double> sx(n, 0), sy(n, 0);
  for (int i = 0; i < n; ++i) {
    S->rooms[i].id = uint32_t(i + 1);
    S->rooms[i].bmin[0] = S->rooms[i].bmin[1] = 1e18;
    S->rooms[i].bmax[0] = S->rooms[i].bmax[1] = -1e18;
  }
  S->ids.assign(N, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t c = size_t(y) * W + x;
      const int v = lab[c];
      if (!v) continue;
      S->ids[c] = uint32_t(v);
      RoomGeom& r = S->rooms[v - 1];
      r.n_cells++;
      sx[v - 1] += x;
      sy[v - 1] += y;
      r.bmin[0] = std::min(r.bmin[0], double(x));
      r.bmin[1] = std::min(r.bmin[1], double(y));
      r.bmax[0] = std::max(r.bmax[0], double(x));
      r.bmax[1] = std::max(r.bmax[1], double(y));
      r.max_clear = std::max(r.max_clear, double(clear[c]));
    }
  for (int i = 0; i < n; ++i) {
    RoomGeom& r = S->rooms[i];
    r.area_m2 = r.n_cells * a1;
    r.centroid[0] = (g.gx0 + sx[i] / std::max(1, r.n_cells) + 0.5) * res;
    r.centroid[1] = (g.gy0 + sy[i] / std::max(1, r.n_cells) + 0.5) * res;
    for (int k = 0; k < 2; ++k) {
      r.bmin[k] = ((k ? g.gy0 : g.gx0) + r.bmin[k]) * res;
      r.bmax[k] = ((k ? g.gy0 : g.gx0) + r.bmax[k] + 1) * res;
    }
  }
  {   // 이음매를 새 번호로 모음(다시 세는 것과 같음: 합친 방 안쪽 이음매는 사라짐)
    std::unordered_map<uint64_t, Seam> m2;
    for (auto& [k, q] : sm) {
      const int a = remap[int(k >> 32)], b = remap[int(k & 0xffffffffu)];
      if (!a || !b || a == b) continue;
      Seam& t = m2[(uint64_t(std::min(a, b)) << 32) | uint32_t(std::max(a, b))];
      t.n += q.n; t.sx += q.sx; t.sy += q.sy; t.cmax = std::max(t.cmax, q.cmax);
    }
    sm.swap(m2);
  }
  const int min_seam = 2;
  for (auto& [k, s] : sm) {
    if (s.n < min_seam) continue;
    RoomDoor d;
    d.a = uint32_t(k >> 32);
    d.b = uint32_t(k & 0xffffffffu);
    d.pos[0] = (g.gx0 + s.sx / s.n + 0.5) * res;
    d.pos[1] = (g.gy0 + s.sy / s.n + 0.5) * res;
    d.width = 2 * s.cmax + res;
    d.seam = s.n;
    S->doors.push_back(d);
  }
  std::sort(S->doors.begin(), S->doors.end(), [](const RoomDoor& x, const RoomDoor& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
  S->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return S;
}

void matchRoomIds(RoomSeg& cur, const RoomSeg* prev, uint32_t* next_id, double match_min) {
  const int n = int(cur.rooms.size());
  std::vector<uint32_t> nid(n + 1, 0);   // 지금 id(1..n) → 새 id
  if (prev && !prev->rooms.empty() && !cur.ids.empty()) {
    const int m = int(prev->rooms.size());
    std::unordered_map<uint32_t, int> pidx;
    for (int j = 0; j < m; ++j) pidx[prev->rooms[j].id] = j;
    std::vector<int> ov(size_t(n) * m, 0);
    const int dx = cur.gx0 - prev->gx0, dy = cur.gy0 - prev->gy0;
    for (int y = 0; y < cur.h; ++y) {
      const int py = y + dy;
      if (py < 0 || py >= prev->h) continue;
      for (int x = 0; x < cur.w; ++x) {
        const uint32_t a = cur.ids[size_t(y) * cur.w + x];
        if (!a) continue;
        const int px = x + dx;
        if (px < 0 || px >= prev->w) continue;
        const uint32_t b = prev->ids[size_t(py) * prev->w + px];
        if (!b) continue;
        auto it = pidx.find(b);
        if (it != pidx.end()) ov[size_t(a - 1) * m + it->second]++;
      }
    }
    std::vector<std::tuple<int, int, int>> pairs;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j) {
        const int o = ov[size_t(i) * m + j];
        if (o > 0 && o >= match_min * std::min(cur.rooms[i].n_cells, prev->rooms[j].n_cells)) pairs.push_back({o, i, j});
      }
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<uint8_t> usedc(n, 0), usedp(m, 0);
    for (auto& [o, i, j] : pairs) {
      if (usedc[i] || usedp[j]) continue;
      usedc[i] = usedp[j] = 1;
      nid[i + 1] = prev->rooms[j].id;
    }
  }
  for (int i = 1; i <= n; ++i)
    if (!nid[i]) nid[i] = (*next_id)++;
  for (uint32_t& v : cur.ids) v = v ? nid[v] : 0;
  for (RoomGeom& r : cur.rooms) r.id = nid[r.id];
  for (RoomDoor& d : cur.doors) {
    d.a = nid[d.a];
    d.b = nid[d.b];
    if (d.a > d.b) std::swap(d.a, d.b);
  }
  std::sort(cur.rooms.begin(), cur.rooms.end(), [](const RoomGeom& a, const RoomGeom& b) { return a.id < b.id; });
  std::sort(cur.doors.begin(), cur.doors.end(), [](const RoomDoor& x, const RoomDoor& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
}

// ---- 물체 배정·이름 ----

uint32_t assignObject(const RoomSeg& s, const RoomObj& o, const RoomParams& p) {
  if (s.ids.empty()) return 0;
  const double res = s.res;
  const double hx = std::max(0.1, o.ext[0] / 2) + p.footprint_margin, hy = std::max(0.1, o.ext[1] / 2) + p.footprint_margin;
  const int x0 = std::max(0, int(std::floor((o.pos[0] - hx) / res)) - s.gx0), x1 = std::min(s.w - 1, int(std::floor((o.pos[0] + hx) / res)) - s.gx0);
  const int y0 = std::max(0, int(std::floor((o.pos[1] - hy) / res)) - s.gy0), y1 = std::min(s.h - 1, int(std::floor((o.pos[1] + hy) / res)) - s.gy0);
  std::unordered_map<uint32_t, int> votes;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
      if (const uint32_t v = s.ids[size_t(y) * s.w + x]) votes[v]++;
  uint32_t best = 0;
  int bn = 0;
  for (auto& [id, n] : votes)
    if (n > bn || (n == bn && id < best)) { bn = n; best = id; }
  if (best) return best;
  // 가까운 방 칸(고리 넓히기)
  const int cx = int(std::floor(o.pos[0] / res)) - s.gx0, cy = int(std::floor(o.pos[1] / res)) - s.gy0;
  const int R = int(std::ceil(p.obj_search_m / res));
  double bd = 1e18;
  for (int r = 1; r <= R; ++r) {
    for (int y = cy - r; y <= cy + r; ++y)
      for (int x = cx - r; x <= cx + r; ++x) {
        if (std::max(std::abs(x - cx), std::abs(y - cy)) != r || x < 0 || y < 0 || x >= s.w || y >= s.h) continue;
        const uint32_t v = s.ids[size_t(y) * s.w + x];
        const double d = double(x - cx) * (x - cx) + double(y - cy) * (y - cy);
        if (v && d < bd && d <= double(R) * R) { bd = d; best = v; }
      }
    if (best && bd <= double(r) * r) break;   // 다음 고리는 더 멀다
  }
  return best;
}

namespace {
struct Rule { const char* key; const char* type; double w; };
// 머리 명사 규칙(이름 == key 이거나 " key" 로 끝남, 끝 's' 무시). 한 물체가 여러 규칙에 맞으면 가장 긴 key
const Rule kRules[] = {
    {"refrigerator", "kitchen", 3}, {"fridge", "kitchen", 3}, {"oven", "kitchen", 3}, {"stove", "kitchen", 3},
    {"cooktop", "kitchen", 3},      {"range hood", "kitchen", 2}, {"microwave", "kitchen", 2}, {"dishwasher", "kitchen", 3},
    {"toaster", "kitchen", 2},      {"kettle", "kitchen", 1},  {"coffee maker", "kitchen", 1}, {"kitchen sink", "kitchen", 3},
    {"sink", "kitchen", 1},         {"sink", "bathroom", 1},
    {"toilet", "bathroom", 3},      {"bathtub", "bathroom", 3}, {"bath", "bathroom", 2}, {"shower", "bathroom", 3},
    {"bathroom sink", "bathroom", 3},
    {"bed", "bedroom", 3},          {"nightstand", "bedroom", 2}, {"wardrobe", "bedroom", 1}, {"dresser", "bedroom", 1},
    {"pillow", "bedroom", 1},
    {"sofa", "living room", 3},     {"couch", "living room", 3}, {"tv", "living room", 2}, {"television", "living room", 2},
    {"coffee table", "living room", 2}, {"armchair", "living room", 1}, {"fireplace", "living room", 1},
    {"desk", "office", 2},          {"monitor", "office", 2},  {"office chair", "office", 2}, {"computer", "office", 1},
    {"keyboard", "office", 1},      {"printer", "office", 1},
};
constexpr double kPrior = 1.5;     // "unknown" 몫(점수 단위)
constexpr double kNameMin = 2.0;   // 이 점수 이상이어야 종류 이름

std::string norm(std::string t) {
  const size_t p = t.find(".n.");
  if (p != std::string::npos) t.resize(p);
  for (char& ch : t) ch = ch == '_' ? ' ' : char(std::tolower(static_cast<unsigned char>(ch)));
  while (!t.empty() && t.back() == ' ') t.pop_back();
  while (!t.empty() && t.front() == ' ') t.erase(t.begin());
  return t;
}
bool head(const std::string& n, const std::string& k) {
  auto ends = [](const std::string& a, const std::string& b) {
    return a == b || (a.size() > b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0 && a[a.size() - b.size() - 1] == ' ');
  };
  return ends(n, k) || (n.size() > 1 && n.back() == 's' && ends(n.substr(0, n.size() - 1), k));
}
}  // namespace

RoomNaming nameRooms(const RoomSeg& s, const std::vector<RoomObj>& objs, const RoomParams& p,
                     const std::unordered_map<uint32_t, NameOverride>* ov) {
  RoomNaming out;
  out.rooms.resize(s.rooms.size());
  out.obj_room.resize(objs.size(), 0);
  std::vector<std::map<std::string, double>> score(s.rooms.size());
  for (size_t i = 0; i < objs.size(); ++i) {
    const uint32_t rid = assignObject(s, objs[i], p);
    out.obj_room[i] = rid;
    const int ri = rid ? s.index(rid) : -1;
    if (ri < 0) continue;
    RoomLabel& L = out.rooms[ri];
    L.objects.push_back(objs[i].id);
    const std::string n = norm(objs[i].name);
    size_t best_len = 0;
    for (const Rule& r : kRules)
      if (head(n, r.key)) best_len = std::max(best_len, std::strlen(r.key));
    if (!best_len) continue;
    for (const Rule& r : kRules)
      if (std::strlen(r.key) == best_len && head(n, r.key)) {
        score[ri][r.type] += r.w;
        L.evidence.push_back({objs[i].id, objs[i].name, r.type, r.w});
      }
  }
  std::map<std::string, int> used;
  for (size_t i = 0; i < s.rooms.size(); ++i) {
    RoomLabel& L = out.rooms[i];
    L.id = s.rooms[i].id;
    double tot = kPrior, best = 0;
    std::string bt;
    for (auto& [t, v] : score[i]) {
      tot += v;
      if (v > best) { best = v; bt = t; }
    }
    for (auto& [t, v] : score[i]) L.probs[t] = v / tot;
    L.probs["unknown"] = kPrior / tot;
    if (ov) {
      auto it = ov->find(L.id);
      if (it != ov->end()) {
        L.name = it->second.name;
        L.type = it->second.name;
        L.conf = it->second.conf;
        L.external = true;
        continue;
      }
    }
    if (best >= kNameMin) {
      L.type = bt;
      const int k = ++used[bt];
      L.name = k == 1 ? bt : bt + " " + std::to_string(k);
      L.conf = float(best / tot);
    } else {
      L.name = "room " + std::to_string(L.id);
      L.conf = float(best / tot);   // 이름 없는 방: 가장 그럴듯한 종류의 확률(근거 없으면 0)
    }
  }
  return out;
}

// ---- 추적 ----

void RoomTracker::setParams(const RoomParams& p) {
  std::lock_guard<std::mutex> g(mu_);
  p_ = p;
  last_check_ = -1e18;   // 다음 update 에서 다시 봄
}
RoomParams RoomTracker::params() const {
  std::lock_guard<std::mutex> g(mu_);
  return p_;
}
std::shared_ptr<const RoomSeg> RoomTracker::current() const {
  std::lock_guard<std::mutex> g(mu_);
  return cur_;
}
int RoomTracker::n_runs() const {
  std::lock_guard<std::mutex> g(mu_);
  return runs_;
}
void RoomTracker::setName(uint32_t id, const char* name, float conf) {
  std::lock_guard<std::mutex> g(mu_);
  if (!name) ov_.erase(id);
  else ov_[id] = NameOverride{name, conf};
}
std::unordered_map<uint32_t, NameOverride> RoomTracker::overrides() const {
  std::lock_guard<std::mutex> g(mu_);
  return ov_;
}
void RoomTracker::reset() {
  std::lock_guard<std::mutex> g(mu_);
  cur_.reset();
  next_id_ = 1;
  epoch_++;
  last_check_ = -1e18;
  ov_.clear();
}

std::shared_ptr<const RoomSeg> RoomTracker::update(const GridView& g, double stamp, bool force) {
  std::shared_ptr<const RoomSeg> prev;
  RoomParams P;
  uint32_t next;
  uint64_t ep;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!p_.enabled || busy_ || !g.cells || g.w <= 0 || g.h <= 0) return cur_;
    const bool back = cur_ && stamp < last_check_ - 1e-9;   // 시각이 뒤로(새 판 등)
    if (!force && cur_ && !back && stamp - last_check_ < p_.period_s) return cur_;
    last_check_ = stamp;
    busy_ = true;
    prev = cur_;
    P = p_;
    next = next_id_;
    ep = epoch_;
  }
  bool run = force || !prev;
  if (!run) {   // 빈칸 분류가 얼마나 바뀌었나(넓어진 곳 포함)
    const size_t lim = size_t(std::max(1.0, P.min_change_m2 / (g.res * g.res)));
    size_t diff = 0;
    const int dx = g.gx0 - prev->gx0, dy = g.gy0 - prev->gy0;
    for (int y = 0; y < g.h && diff < lim; ++y)
      for (int x = 0; x < g.w; ++x) {
        const int v = g.cells[size_t(y) * g.w + x];
        const uint8_t f = v >= 0 && v <= P.free_max;
        const int px = x + dx, py = y + dy;
        const uint8_t pf = (px >= 0 && py >= 0 && px < prev->w && py < prev->h) ? prev->rawfree[size_t(py) * prev->w + px] : 0;
        diff += f != pf;
      }
    run = diff >= lim || std::abs(g.res - prev->res) > 1e-9;
  }
  std::shared_ptr<RoomSeg> s;
  if (run) {
    s = segmentRooms(g, P);
    s->stamp = stamp;
    const auto t0 = std::chrono::steady_clock::now();
    matchRoomIds(*s, prev.get(), &next, P.match_min);
    s->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::lock_guard<std::mutex> lk(mu_);
  busy_ = false;
  if (s && ep == epoch_) {
    cur_ = std::move(s);
    next_id_ = next;
    runs_++;
  }
  return cur_;
}

}  // namespace scenemap
