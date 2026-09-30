// 섬 관리자 상태 넘겨받기 (문서 15.3 v1-b, 리드): solver 의 섬 관리(core/solver/islands.h, PhysX IslandSim·SimpleIslandManager 손 구현)를
// 에피소드 경계에서 통째로 채우기 위한 "배열 그대로" 상태와 파일 읽기·쓰기, 그리고 그것을 ig::IslandManager 로 세우는 저장소.
// 번호(노드·간선·섬 번호, 사슬 순서, 활성 목록 순서)는 PhysX 와 같은 값을 그대로 둔다 — 풀이 순서의 뿌리라서.
// 우선순위 큐·방문 목록은 스텝 사이에 비어 있으므로 넘기지 않는다(용량만).
#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/solver/islands.h"

namespace eng {
namespace scene {

struct HandleState {
  std::vector<uint32_t> freeHandles;
  uint32_t currentHandle = 0;
};

struct IslandSimState {
  HandleState islandHandles;
  std::vector<ig::Node> nodes;
  std::vector<uint32_t> activeNodeIndex, hopCounts, fastRoute, islandIds;
  std::vector<ig::Edge> edges;
  std::vector<ig::EdgeInstance> edgeInstances;
  std::vector<ig::Island> islands;
  std::vector<uint32_t> islandStaticTouchCount;
  std::vector<uint32_t> activeNodes[ig::NODE_TYPES], nodesToPutToSleep[ig::NODE_TYPES];
  std::vector<uint32_t> activeKinematicNodes;
  std::vector<uint32_t> activatedEdges[ig::EDGE_TYPES], dirtyEdges[ig::EDGE_TYPES], islandSplitEdges[ig::EDGE_TYPES], deactivatingEdges[ig::EDGE_TYPES];
  uint32_t activeEdgeCount[ig::EDGE_TYPES] = {}, initialActiveNodeCount[ig::EDGE_TYPES] = {};
  std::vector<uint32_t> islandAwake, dirtyMap;  // 비트 낱말 (PhysX PxBitMap 낱말 수 그대로)
  std::vector<uint32_t> activeIslands, activatingNodes, destroyedEdges;
};

struct IslandMgrState {
  bool valid = false;
  HandleState nodeHandles, edgeHandles;
  std::vector<uint32_t> destroyedNodes, destroyedEdges;
  std::vector<ig::NodeIndex> edgeNodeIndices;  // 2 * 간선
  std::vector<uint32_t> constraintOrCm;       // 간선 -> 관리자/제약 (장면 파일 번호, 없으면 INVALID)
  std::vector<uint32_t> connectedMap;         // 비트 낱말
  IslandSimState accurate, speculative;
};

// ---- 파일 (개수 앞에 붙인 배열들)
namespace isio {
template <class T>
inline bool w(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  return fwrite(&n, 8, 1, f) == 1 && (n == 0 || fwrite(v.data(), sizeof(T), n, f) == n);
}
template <class T>
inline bool r(FILE* f, std::vector<T>& v) {
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1 || n > (1ull << 32)) return false;
  v.resize(n);
  return n == 0 || fread(v.data(), sizeof(T), n, f) == n;
}
inline bool wh(FILE* f, const HandleState& h) { return w(f, h.freeHandles) && fwrite(&h.currentHandle, 4, 1, f) == 1; }
inline bool rh(FILE* f, HandleState& h) { return r(f, h.freeHandles) && fread(&h.currentHandle, 4, 1, f) == 1; }
template <class F, class H, class V>
inline bool sim(FILE* f, IslandSimState& s, F vec, H hnd, V raw) {
  bool ok = hnd(f, s.islandHandles) && vec(f, s.nodes) && vec(f, s.activeNodeIndex) && vec(f, s.hopCounts) && vec(f, s.fastRoute) && vec(f, s.islandIds) &&
            vec(f, s.edges) && vec(f, s.edgeInstances) && vec(f, s.islands) && vec(f, s.islandStaticTouchCount) && vec(f, s.activeKinematicNodes) &&
            vec(f, s.islandAwake) && vec(f, s.dirtyMap) && vec(f, s.activeIslands) && vec(f, s.activatingNodes) && vec(f, s.destroyedEdges);
  for (uint32_t t = 0; ok && t < ig::NODE_TYPES; ++t) ok = vec(f, s.activeNodes[t]) && vec(f, s.nodesToPutToSleep[t]);
  for (uint32_t t = 0; ok && t < ig::EDGE_TYPES; ++t)
    ok = vec(f, s.activatedEdges[t]) && vec(f, s.dirtyEdges[t]) && vec(f, s.islandSplitEdges[t]) && vec(f, s.deactivatingEdges[t]);
  return ok && raw(f, s.activeEdgeCount, sizeof(s.activeEdgeCount)) && raw(f, s.initialActiveNodeCount, sizeof(s.initialActiveNodeCount));
}
}  // namespace isio

inline bool writeIslands(FILE* f, IslandMgrState& s) {
  auto vec = [](FILE* ff, auto& v) { return isio::w(ff, v); };
  auto raw = [](FILE* ff, void* p, size_t n) { return fwrite(p, 1, n, ff) == n; };
  const uint32_t magic = 0x49534c31u;  // "ISL1"
  return fwrite(&magic, 4, 1, f) == 1 && isio::wh(f, s.nodeHandles) && isio::wh(f, s.edgeHandles) && isio::w(f, s.destroyedNodes) &&
         isio::w(f, s.destroyedEdges) && isio::w(f, s.edgeNodeIndices) && isio::w(f, s.constraintOrCm) && isio::w(f, s.connectedMap) &&
         isio::sim(f, s.accurate, vec, isio::wh, raw) && isio::sim(f, s.speculative, vec, isio::wh, raw);
}
inline bool readIslands(FILE* f, IslandMgrState& s) {
  auto vec = [](FILE* ff, auto& v) { return isio::r(ff, v); };
  auto raw = [](FILE* ff, void* p, size_t n) { return fread(p, 1, n, ff) == n; };
  uint32_t magic = 0;
  s.valid = fread(&magic, 4, 1, f) == 1 && magic == 0x49534c31u && isio::rh(f, s.nodeHandles) && isio::rh(f, s.edgeHandles) &&
            isio::r(f, s.destroyedNodes) && isio::r(f, s.destroyedEdges) && isio::r(f, s.edgeNodeIndices) && isio::r(f, s.constraintOrCm) &&
            isio::r(f, s.connectedMap) && isio::sim(f, s.accurate, vec, isio::rh, raw) && isio::sim(f, s.speculative, vec, isio::rh, raw);
  return s.valid;
}

// ---- ig::IslandManager 저장소 (용량 고정 배열을 벡터로 소유). 판 N 개 GPU 판에서는 같은 배치를 판마다 이어 붙인다(G2).
struct IslandStore {
  std::vector<std::vector<uint32_t>> u32;
  std::vector<std::vector<ig::Node>> nodes;
  std::vector<std::vector<ig::Edge>> edges;
  std::vector<std::vector<ig::EdgeInstance>> einst;
  std::vector<std::vector<ig::Island>> islands;
  std::vector<std::vector<ig::QueueElement>> pqs;
  std::vector<std::vector<ig::TraversalState>> vis;
  std::vector<ig::NodeIndex> eni;
  ig::IslandManager M;
  uint32_t* u(uint32_t n) {
    u32.emplace_back(n, 0u);
    return u32.back().data();
  }
  ig::Arr<uint32_t> au(uint32_t n) { return ig::Arr<uint32_t>{u(n), 0, n}; }
  ig::Bits bits(uint32_t nbits) { return ig::Bits{u((nbits + 31) / 32), 0, (nbits + 31) / 32}; }
  void initSim(ig::IslandSim& S, uint32_t CN, uint32_t CE) {
    S = ig::IslandSim{};
    S.islandHandles = ig::HandleManager{au(CN + 8), 0};
    nodes.emplace_back(CN);
    S.nodes = ig::Arr<ig::Node>{nodes.back().data(), 0, CN};
    S.activeNodeIndex = u(CN);
    S.hopCounts = u(CN);
    S.fastRoute = u(CN);
    S.islandIds = u(CN);
    edges.emplace_back(CE);
    S.edges = ig::Arr<ig::Edge>{edges.back().data(), 0, CE};
    einst.emplace_back(2 * CE);
    S.edgeInstances = ig::Arr<ig::EdgeInstance>{einst.back().data(), 0, 2 * CE};
    islands.emplace_back(CN + 8);
    S.islands = ig::Arr<ig::Island>{islands.back().data(), 0, CN + 8};
    S.islandStaticTouchCount = u(CN + 8);
    for (uint32_t t = 0; t < ig::NODE_TYPES; ++t) {
      S.activeNodes[t] = au(CN);
      S.nodesToPutToSleep[t] = au(CN);
    }
    S.activeKinematicNodes = au(CN);
    for (uint32_t t = 0; t < ig::EDGE_TYPES; ++t) {
      S.activatedEdges[t] = au(CE);
      S.dirtyEdges[t] = au(CE);
      S.islandSplitEdges[t] = au(CE + 1);
      S.deactivatingEdges[t] = au(CE);
    }
    S.islandAwake = bits(CN + 64);
    S.activeIslands = au(CN + 8);
    S.dirtyMap = bits(CN + 64);
    S.activatingNodes = au(CN);
    S.destroyedEdges = au(CE);
    pqs.emplace_back(CN + 8);
    S.pq = ig::Arr<ig::QueueElement>{pqs.back().data(), 0, CN + 8};
    vis.emplace_back(CN + 8);
    S.visitedNodes = ig::Arr<ig::TraversalState>{vis.back().data(), 0, CN + 8};
    S.visitedState = bits(CN + 64);
    S.cpu = &M.cpu;
  }
  void init(uint32_t CN, uint32_t CE) {
    u32.clear(); nodes.clear(); edges.clear(); einst.clear(); islands.clear(); pqs.clear(); vis.clear();
    M = ig::IslandManager{};
    M.nodeHandles = ig::HandleManager{au(CN), 0};
    M.edgeHandles = ig::HandleManager{au(CE), 0};
    M.destroyedNodes = au(CN);
    M.destroyedEdges = au(CE);
    eni.assign(2 * CE, ig::NodeIndex{ig::INVALID_NODE, 0});
    M.cpu = ig::CpuData{eni.data(), 2 * CE};
    M.constraintOrCm = u(CE);
    M.connectedMap = bits(CE + 64);
    initSim(M.accurate, CN, CE);
    initSim(M.speculative, CN, CE);
  }

  // 상태 -> 저장소. 용량 = 상태 크기 x2 + 여유. 넘치면 false.
  bool load(const IslandMgrState& s) {
    uint32_t CN = uint32_t(s.accurate.nodes.size()), CE = uint32_t(s.edgeNodeIndices.size() / 2);
    CN = CN * 2 + 1024;
    CE = CE * 2 + 4096;
    init(CN, CE);
    bool ok = true;
    auto fillA = [&](ig::Arr<uint32_t>& a, const std::vector<uint32_t>& v) {
      if (v.size() > a.cap) { ok = false; return; }
      for (size_t k = 0; k < v.size(); ++k) a.d[k] = v[k];
      a.size = uint32_t(v.size());
    };
    auto fillH = [&](ig::HandleManager& h, const HandleState& v) {
      fillA(h.freeHandles, v.freeHandles);
      h.currentHandle = v.currentHandle;
    };
    auto fillB = [&](ig::Bits& b, const std::vector<uint32_t>& v) {
      if (v.size() > b.capWords) { ok = false; return; }
      for (size_t k = 0; k < v.size(); ++k) b.w[k] = v[k];
      b.words = uint32_t(v.size());
    };
    auto fillP = [&](uint32_t* p, const std::vector<uint32_t>& v, uint32_t cap) {
      if (v.size() > cap) { ok = false; return; }
      for (size_t k = 0; k < v.size(); ++k) p[k] = v[k];
    };
    fillH(M.nodeHandles, s.nodeHandles);
    fillH(M.edgeHandles, s.edgeHandles);
    fillA(M.destroyedNodes, s.destroyedNodes);
    fillA(M.destroyedEdges, s.destroyedEdges);
    if (s.edgeNodeIndices.size() > eni.size()) return false;
    for (size_t k = 0; k < s.edgeNodeIndices.size(); ++k) eni[k] = s.edgeNodeIndices[k];
    fillP(M.constraintOrCm, s.constraintOrCm, CE);
    for (uint32_t k = uint32_t(s.constraintOrCm.size()); k < CE; ++k) M.constraintOrCm[k] = ig::INVALID_EDGE;
    fillB(M.connectedMap, s.connectedMap);
    auto fillSim = [&](ig::IslandSim& S, const IslandSimState& v) {
      fillH(S.islandHandles, v.islandHandles);
      if (v.nodes.size() > S.nodes.cap || v.edges.size() > S.edges.cap || v.edgeInstances.size() > S.edgeInstances.cap || v.islands.size() > S.islands.cap) {
        ok = false;
        return;
      }
      for (size_t k = 0; k < v.nodes.size(); ++k) S.nodes.d[k] = v.nodes[k];
      S.nodes.size = uint32_t(v.nodes.size());
      fillP(S.activeNodeIndex, v.activeNodeIndex, S.nodes.cap);
      fillP(S.hopCounts, v.hopCounts, S.nodes.cap);
      fillP(S.fastRoute, v.fastRoute, S.nodes.cap);
      fillP(S.islandIds, v.islandIds, S.nodes.cap);
      for (size_t k = 0; k < v.edges.size(); ++k) S.edges.d[k] = v.edges[k];
      S.edges.size = uint32_t(v.edges.size());
      for (size_t k = 0; k < v.edgeInstances.size(); ++k) S.edgeInstances.d[k] = v.edgeInstances[k];
      S.edgeInstances.size = uint32_t(v.edgeInstances.size());
      for (size_t k = 0; k < v.islands.size(); ++k) S.islands.d[k] = v.islands[k];
      S.islands.size = uint32_t(v.islands.size());
      fillP(S.islandStaticTouchCount, v.islandStaticTouchCount, S.islands.cap);
      for (uint32_t t = 0; t < ig::NODE_TYPES; ++t) {
        fillA(S.activeNodes[t], v.activeNodes[t]);
        fillA(S.nodesToPutToSleep[t], v.nodesToPutToSleep[t]);
      }
      fillA(S.activeKinematicNodes, v.activeKinematicNodes);
      for (uint32_t t = 0; t < ig::EDGE_TYPES; ++t) {
        fillA(S.activatedEdges[t], v.activatedEdges[t]);
        fillA(S.dirtyEdges[t], v.dirtyEdges[t]);
        fillA(S.islandSplitEdges[t], v.islandSplitEdges[t]);
        fillA(S.deactivatingEdges[t], v.deactivatingEdges[t]);
        S.activeEdgeCount[t] = v.activeEdgeCount[t];
        S.initialActiveNodeCount[t] = v.initialActiveNodeCount[t];
      }
      fillB(S.islandAwake, v.islandAwake);
      fillA(S.activeIslands, v.activeIslands);
      fillB(S.dirtyMap, v.dirtyMap);
      fillA(S.activatingNodes, v.activatingNodes);
      fillA(S.destroyedEdges, v.destroyedEdges);
    };
    fillSim(M.accurate, s.accurate);
    fillSim(M.speculative, s.speculative);
    return ok;
  }
};

}  // namespace scene
}  // namespace eng
