// 섬 관리 (손으로 짬, PhysX 5.6.1 과 같은 순서): 노드(몸체)·간선(접촉 관리자·제약)으로 섬을 만들고 합치고 쪼개며,
// 깨우기·재우기, 활성 섬 목록·섬 안 노드 사슬·간선 사슬 순서를 PhysX 와 똑같이 만든다(풀이 순서의 뿌리).
// 원본 (physx/source/lowlevel/software/)
//   include/PxsIslandSim.h:119-854   HandleManager / Node / Edge / Island / 인라인 함수(markActive·markInactive·markEdgeActive·…)
//   src/PxsIslandSim.cpp:77-2175     IslandSim 전부 (addNode, addConnection, wakeIslands, processNewEdges, processLostEdges,
//                                    findRoute(우선순위 큐), mergeIslands, setKinematic/setDynamic …)
//   src/PxsSimpleIslandManager.cpp   두 섬 시뮬(추측 speculative / 정확 accurate)과 노드·간선 번호 관리, 1·2·3차 섬 생성
//   common/src/CmPriorityQueue.h     이진 힙 (같은 hop 수에서 꺼내는 순서가 결과를 정한다)
// 자료는 고정 용량 배열(번호로 잇기) — 넘치면 overflow. 호스트·GPU 공용(SV_HD).
#pragma once
#include <cstdint>

#include "saos.h"

namespace eng {
namespace ig {

constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;  // PX_INVALID_NODE
constexpr uint32_t INVALID_EDGE = 0xFFFFFFFFu;  // IG_INVALID_EDGE
constexpr uint32_t INVALID_ISLAND = 0xFFFFFFFFu;
constexpr uint32_t NODE_TYPES = 5;  // Node::eTYPE_COUNT
constexpr uint32_t EDGE_TYPES = 5;  // Edge::eEDGE_TYPE_COUNT

enum NodeType : uint8_t { eRIGID_BODY_TYPE = 0, eARTICULATION_TYPE = 1 };
enum EdgeType : uint16_t { eCONTACT_MANAGER = 0, eCONSTRAINT = 1 };

enum : uint16_t {  // Edge::EdgeState
  E_INSERTED = 1 << 0,
  E_PENDING_DESTROYED = 1 << 1,
  E_ACTIVE = 1 << 2,
  E_IN_DIRTY_LIST = 1 << 3,
  E_DESTROYED = 1 << 4,
  E_REPORT_ONLY_DESTROY = 1 << 5,
  E_ACTIVATING = 1 << 6,
};
enum : uint8_t {  // Node::State
  N_READY_FOR_SLEEPING = 1u << 0,
  N_ACTIVE = 1u << 1,
  N_KINEMATIC = 1u << 2,
  N_DELETED = 1u << 3,
  N_DIRTY = 1u << 4,
  N_ACTIVATING = 1u << 5,
};

// PxNodeIndex (index + 관절체 링크 번호). 섬 관리는 index 만 쓴다
struct NodeIndex {
  uint32_t id, linkId;
  SV_HD uint32_t index() const { return id; }
  SV_HD bool isValid() const { return id != INVALID_NODE; }
  SV_HD bool isStaticBody() const { return id == INVALID_NODE; }
};
SV_HD NodeIndex nidx(uint32_t i) { return NodeIndex{i, 0u}; }

struct Edge {
  uint16_t type;
  uint16_t state;
  uint32_t nextIslandEdge, prevIslandEdge;
};
SV_HD Edge edgeDefault() { return Edge{eCONTACT_MANAGER, E_DESTROYED, INVALID_EDGE, INVALID_EDGE}; }
struct EdgeInstance {
  uint32_t next, prev;
};
struct Node {
  uint32_t firstEdgeIndex;
  uint8_t flags, type;
  uint16_t staticTouchCount;
  uint32_t nextNode, prevNode;  // 노드 번호 (index)
  uint32_t activeRefCount;
  uint32_t object;              // 몸체 번호 등 (void* 대신)
};
SV_HD Node nodeDefault() { return Node{INVALID_EDGE, N_DELETED, eRIGID_BODY_TYPE, 0, INVALID_NODE, INVALID_NODE, 0, 0}; }
struct Island {
  uint32_t rootNode, lastNode;
  uint32_t nodeCount[NODE_TYPES];
  uint32_t activeIndex;
  uint32_t firstEdge[EDGE_TYPES], lastEdge[EDGE_TYPES];
  uint32_t edgeCount[EDGE_TYPES];
};
SV_HD Island islandDefault() {
  Island is;
  is.rootNode = is.lastNode = INVALID_NODE;
  is.activeIndex = INVALID_ISLAND;
  for (uint32_t a = 0; a < EDGE_TYPES; ++a) {
    is.firstEdge[a] = is.lastEdge[a] = INVALID_EDGE;
    is.edgeCount[a] = 0;
  }
  for (uint32_t a = 0; a < NODE_TYPES; ++a) is.nodeCount[a] = 0;
  return is;
}
struct TraversalState {
  uint32_t nodeIndex, currentIndex, prevIndex, depth;
};
struct QueueElement {  // mState 는 mVisitedNodes 번호 (원본은 포인터)
  uint32_t state, hopCount;
};

// ---- 고정 용량 배열 도우미
template <class T>
struct Arr {
  T* d;
  uint32_t size, cap;
  SV_HD T& operator[](uint32_t i) { return d[i]; }
  SV_HD const T& operator[](uint32_t i) const { return d[i]; }
  SV_HD bool push(const T& v, uint32_t& err) {
    if (size >= cap) {
      err |= 1u;
      return false;
    }
    d[size++] = v;
    return true;
  }
  SV_HD T& back() { return d[size - 1]; }
  SV_HD T popBack() { return d[--size]; }
  SV_HD void replaceWithLast(uint32_t i) { d[i] = d[--size]; }
};
struct Bits {  // PxBitMap (32비트 낱말)
  uint32_t* w;
  uint32_t words;  // 쓰는 낱말 수 (growAndSet 으로 늘어남), 용량은 capWords
  uint32_t capWords;
  SV_HD bool test(uint32_t i) const { return (i >> 5) < words && ((w[i >> 5] >> (i & 31)) & 1u); }
  SV_HD void set(uint32_t i) { w[i >> 5] |= (1u << (i & 31)); }
  SV_HD void reset(uint32_t i) { if ((i >> 5) < words) w[i >> 5] &= ~(1u << (i & 31)); }
  SV_HD void ensure(uint32_t i, uint32_t& err) {  // growAndSet/growAndReset 의 늘리기
    const uint32_t need = (i >> 5) + 1;
    if (need > words) {
      if (need > capWords) {
        err |= 2u;
        return;
      }
      for (uint32_t k = words; k < need; ++k) w[k] = 0;
      words = need;
    }
  }
  SV_HD void growAndSet(uint32_t i, uint32_t& err) { ensure(i, err); set(i); }
  SV_HD void growAndReset(uint32_t i, uint32_t& err) { ensure(i, err); w[i >> 5] &= ~(1u << (i & 31)); }
  SV_HD void clearAll() { for (uint32_t k = 0; k < words; ++k) w[k] = 0; }
  SV_HD void resizeAndClear(uint32_t nbits, uint32_t& err) {
    const uint32_t need = (nbits + 31) >> 5;
    if (need > capWords) {
      err |= 2u;
      return;
    }
    words = need;
    for (uint32_t k = 0; k < words; ++k) w[k] = 0;
  }
};

struct HandleManager {  // PxsIslandSim.h:120
  Arr<uint32_t> freeHandles;
  uint32_t currentHandle;
  SV_HD uint32_t getHandle() {
    if (freeHandles.size) return freeHandles.popBack();
    return currentHandle++;
  }
  SV_HD void freeHandle(uint32_t h, uint32_t& err) {
    if (h == currentHandle)  // 원본 그대로: handle == mCurrentHandle 이면 줄임 (실제로는 생기지 않는 경우)
      currentHandle--;
    else
      freeHandles.push(h, err);
  }
};

// 두 섬 시뮬이 함께 보는 간선->노드 표 (CPUExternalData)
struct CpuData {
  NodeIndex* edgeNodeIndices;  // 2 * 간선
  uint32_t cap;
};

struct IslandSim {
  HandleManager islandHandles;
  Arr<Node> nodes;
  uint32_t* activeNodeIndex;
  uint32_t* hopCounts;
  uint32_t* fastRoute;
  uint32_t* islandIds;
  Arr<Edge> edges;
  Arr<EdgeInstance> edgeInstances;
  Arr<Island> islands;
  uint32_t* islandStaticTouchCount;  // islands 와 같은 크기
  Arr<uint32_t> activeNodes[NODE_TYPES];
  Arr<uint32_t> activeKinematicNodes;
  Arr<uint32_t> activatedEdges[EDGE_TYPES];
  uint32_t activeEdgeCount[EDGE_TYPES];
  Bits islandAwake;
  Arr<uint32_t> activeIslands;
  uint32_t initialActiveNodeCount[EDGE_TYPES];
  Arr<uint32_t> nodesToPutToSleep[NODE_TYPES];
  Arr<uint32_t> dirtyEdges[EDGE_TYPES];
  Bits dirtyMap;
  Arr<uint32_t> activatingNodes;
  Arr<uint32_t> destroyedEdges;
  Arr<QueueElement> pq;  // 우선순위 큐 (이진 힙, 크기 = pq.size)
  Arr<TraversalState> visitedNodes;
  Bits visitedState;
  Arr<uint32_t> islandSplitEdges[EDGE_TYPES];
  Arr<uint32_t> deactivatingEdges[EDGE_TYPES];
  const CpuData* cpu;
  uint32_t err;
};

// ---------------- 우선순위 큐 (CmPriorityQueue.h PriorityQueueBase, NodeComparator: hop 작은 것 먼저)
SV_HD bool pqLess(const QueueElement& a, const QueueElement& b) { return a.hopCount < b.hopCount; }
SV_HD void pqPush(IslandSim& S, const QueueElement& value) {
  if (S.pq.size >= S.pq.cap) {
    S.err |= 4u;
    return;
  }
  uint32_t newIndex;
  uint32_t parentIndex = (S.pq.size - 1) >> 1;
  for (newIndex = S.pq.size; newIndex > 0 && pqLess(value, S.pq[parentIndex]); newIndex = parentIndex, parentIndex = (newIndex - 1) >> 1)
    S.pq[newIndex] = S.pq[parentIndex];
  S.pq[newIndex] = value;
  S.pq.size++;
}
SV_HD QueueElement pqPop(IslandSim& S) {
  uint32_t i, child;
  const uint32_t tempHs = S.pq.size - 1;
  S.pq.size = tempHs;
  const QueueElement mn = S.pq[0];
  const QueueElement last = S.pq[tempHs];
  for (i = 0; (child = (i << 1) + 1) < tempHs; i = child) {
    const uint32_t rightChild = child + 1;
    child += ((rightChild < tempHs) & pqLess(S.pq[rightChild], S.pq[child])) ? 1 : 0;
    if (pqLess(last, S.pq[child])) break;
    S.pq[i] = S.pq[child];
  }
  S.pq[i] = last;
  return mn;
}

// ---------------- 인라인 함수들 (PxsIslandSim.h:617-853)
SV_HD void markIslandActive(IslandSim& S, uint32_t islandId) {
  Island& island = S.islands[islandId];
  S.islandAwake.set(islandId);
  island.activeIndex = S.activeIslands.size;
  S.activeIslands.push(islandId, S.err);
}
SV_HD void markIslandInactive(IslandSim& S, uint32_t islandId) {
  Island& island = S.islands[islandId];
  const uint32_t replaceId = S.activeIslands[S.activeIslands.size - 1];
  Island& replaceIsland = S.islands[replaceId];
  replaceIsland.activeIndex = island.activeIndex;
  S.activeIslands[island.activeIndex] = replaceId;
  S.activeIslands.size--;
  island.activeIndex = INVALID_ISLAND;
  S.islandAwake.reset(islandId);
}
SV_HD void markKinematicActive(IslandSim& S, uint32_t index) {
  const Node& node = S.nodes[index];
  if (node.activeRefCount == 0 && S.activeNodeIndex[index] == INVALID_NODE) {
    S.activeNodeIndex[index] = S.activeKinematicNodes.size;
    S.activeKinematicNodes.push(index, S.err);
  }
}
SV_HD void markKinematicInactive(IslandSim& S, uint32_t index) {
  const Node& node = S.nodes[index];
  if (node.activeRefCount == 0) {
    if (S.activeNodeIndex[index] != INVALID_NODE) {
      const uint32_t replaceIndex = S.activeKinematicNodes.back();
      S.activeNodeIndex[replaceIndex] = S.activeNodeIndex[index];
      S.activeKinematicNodes[S.activeNodeIndex[index]] = replaceIndex;
      S.activeKinematicNodes.size--;
      S.activeNodeIndex[index] = INVALID_NODE;
    }
  }
}
SV_HD void markActive(IslandSim& S, uint32_t index) {
  const Node& node = S.nodes[index];
  S.activeNodeIndex[index] = S.activeNodes[node.type].size;
  S.activeNodes[node.type].push(index, S.err);
}
SV_HD void markInactive(IslandSim& S, uint32_t index) {
  const Node& node = S.nodes[index];
  Arr<uint32_t>& activeNodes = S.activeNodes[node.type];
  const uint32_t initialActiveNodeCount = S.initialActiveNodeCount[node.type];
  if (S.activeNodeIndex[index] < initialActiveNodeCount) {
    const uint32_t activeNodeIndex = S.activeNodeIndex[index];
    const uint32_t replaceIndex = activeNodes[initialActiveNodeCount - 1];
    S.activeNodeIndex[index] = S.activeNodeIndex[replaceIndex];
    S.activeNodeIndex[replaceIndex] = activeNodeIndex;
    activeNodes[activeNodeIndex] = replaceIndex;
    activeNodes[S.activeNodeIndex[index]] = index;
    S.initialActiveNodeCount[node.type]--;
  }
  const uint32_t replaceIndex = activeNodes.back();
  S.activeNodeIndex[replaceIndex] = S.activeNodeIndex[index];
  activeNodes[S.activeNodeIndex[index]] = replaceIndex;
  activeNodes.size--;
  S.activeNodeIndex[index] = INVALID_NODE;
}
SV_HD void markEdgeActive(IslandSim& S, uint32_t index, uint32_t index1, uint32_t index2) {
  Edge& edge = S.edges[index];
  edge.state |= E_ACTIVATING;
  S.activatedEdges[edge.type].push(index, S.err);
  S.activeEdgeCount[edge.type]++;
  if (index1 != INVALID_NODE && index2 != INVALID_NODE) {
    {
      Node& node = S.nodes[index1];
      if (node.activeRefCount == 0 && (node.flags & N_KINEMATIC) && !(node.flags & (N_ACTIVE | N_ACTIVATING))) markKinematicActive(S, index1);
      node.activeRefCount++;
    }
    {
      Node& node = S.nodes[index2];
      if (node.activeRefCount == 0 && (node.flags & N_KINEMATIC) && !(node.flags & (N_ACTIVE | N_ACTIVATING))) markKinematicActive(S, index2);
      node.activeRefCount++;
    }
  }
}
SV_HD void removeEdgeFromIsland(IslandSim& S, Island& island, uint32_t edgeIndex) {
  Edge& edge = S.edges[edgeIndex];
  if (edge.nextIslandEdge != INVALID_EDGE)
    S.edges[edge.nextIslandEdge].prevIslandEdge = edge.prevIslandEdge;
  else
    island.lastEdge[edge.type] = edge.prevIslandEdge;
  if (edge.prevIslandEdge != INVALID_EDGE)
    S.edges[edge.prevIslandEdge].nextIslandEdge = edge.nextIslandEdge;
  else
    island.firstEdge[edge.type] = edge.nextIslandEdge;
  island.edgeCount[edge.type]--;
  edge.nextIslandEdge = edge.prevIslandEdge = INVALID_EDGE;
}
SV_HD void addEdgeToIsland(IslandSim& S, Island& island, uint32_t edgeIndex) {
  Edge& edge = S.edges[edgeIndex];
  if (island.lastEdge[edge.type] != INVALID_EDGE)
    S.edges[island.lastEdge[edge.type]].nextIslandEdge = edgeIndex;
  else
    island.firstEdge[edge.type] = edgeIndex;
  edge.prevIslandEdge = island.lastEdge[edge.type];
  island.lastEdge[edge.type] = edgeIndex;
  island.edgeCount[edge.type]++;
}
SV_HD void removeNodeFromIsland(IslandSim& S, Island& island, uint32_t nodeIndex) {
  Node& node = S.nodes[nodeIndex];
  if (node.nextNode != INVALID_NODE)
    S.nodes[node.nextNode].prevNode = node.prevNode;
  else
    island.lastNode = node.prevNode;
  if (node.prevNode != INVALID_NODE)
    S.nodes[node.prevNode].nextNode = node.nextNode;
  else
    island.rootNode = node.nextNode;
  island.nodeCount[node.type]--;
  node.nextNode = node.prevNode = INVALID_NODE;
}

// ---------------- 크기 늘리기 (PxArray::resize 흉내: 새 칸은 기본값)
SV_HD void growNodes(IslandSim& S, uint32_t newSize) {
  if (newSize > S.nodes.cap) {
    S.err |= 8u;
    return;
  }
  for (uint32_t k = S.nodes.size; k < newSize; ++k) {
    S.nodes[k] = nodeDefault();
    S.activeNodeIndex[k] = 0;
    S.hopCounts[k] = 0;
    S.fastRoute[k] = INVALID_NODE;
    S.islandIds[k] = 0;
  }
  if (newSize > S.nodes.size) S.nodes.size = newSize;
}
SV_HD void growIslands(IslandSim& S, uint32_t newSize) {
  if (newSize > S.islands.cap) {
    S.err |= 16u;
    return;
  }
  for (uint32_t k = S.islands.size; k < newSize; ++k) {
    S.islands[k] = islandDefault();
    S.islandStaticTouchCount[k] = 0;
  }
  if (newSize > S.islands.size) S.islands.size = newSize;
}
SV_HD void growEdges(IslandSim& S, uint32_t newSize) {
  if (newSize > S.edges.cap) {
    S.err |= 32u;
    return;
  }
  for (uint32_t k = S.edges.size; k < newSize; ++k) S.edges[k] = edgeDefault();
  if (newSize > S.edges.size) S.edges.size = newSize;
}
SV_HD void growEdgeInstances(IslandSim& S, uint32_t newSize) {
  if (newSize > S.edgeInstances.cap) {
    S.err |= 64u;
    return;
  }
  for (uint32_t k = S.edgeInstances.size; k < newSize; ++k) S.edgeInstances[k] = EdgeInstance{INVALID_EDGE, INVALID_EDGE};
  if (newSize > S.edgeInstances.size) S.edgeInstances.size = newSize;
}

// ---------------- IslandSim (PxsIslandSim.cpp)
SV_HDN void activateNode(IslandSim& S, uint32_t index);

SV_HDN void addNode(IslandSim& S, bool isActive, bool isKinematic, uint8_t type, uint32_t handle, uint32_t object) {
  growNodes(S, handle + 1 > S.nodes.size ? handle + 1 : S.nodes.size);
  if (S.err) return;
  S.activeNodeIndex[handle] = INVALID_NODE;
  Node& node = S.nodes[handle];
  node.type = type;
  uint8_t flags = uint8_t(isActive ? 0 : N_READY_FOR_SLEEPING);
  if (isKinematic) flags |= N_KINEMATIC;
  node.flags = flags;
  S.islandIds[handle] = INVALID_ISLAND;
  S.fastRoute[handle] = INVALID_NODE;
  S.hopCounts[handle] = 0;
  if (!isKinematic) {
    const uint32_t islandHandle = S.islandHandles.getHandle();
    const uint32_t newSize = islandHandle + 1 > S.islands.size ? islandHandle + 1 : S.islands.size;
    growIslands(S, newSize);
    S.islandAwake.growAndReset(newSize, S.err);  // 원본 그대로 mIslandAwake.growAndReset(newSize): 새 크기 칸(섬 밖)을 0 으로
    Island& island = S.islands[islandHandle];
    island.lastNode = island.rootNode = handle;
    island.nodeCount[type] = 1;
    S.islandIds[handle] = islandHandle;
    S.islandStaticTouchCount[islandHandle] = 0;
  }
  if (isActive) activateNode(S, handle);
  node.object = object;
}

SV_HD void preallocateConnections(IslandSim& S, uint32_t handle) { growEdges(S, handle + 1 > S.edges.size ? handle + 1 : S.edges.size); }

SV_HD bool addConnectionPreallocated(IslandSim& S, uint32_t edgeType, uint32_t handle) {
  Edge& edge = S.edges[handle];
  if (edge.state & E_PENDING_DESTROYED) {
    edge.state &= ~E_PENDING_DESTROYED;
    return false;
  }
  if (edge.state & E_IN_DIRTY_LIST) return false;
  edge.state &= ~E_DESTROYED;
  edge.type = uint16_t(edgeType);
  edge.state |= E_IN_DIRTY_LIST;
  edge.state &= ~E_ACTIVATING;
  return true;
}
SV_HD void addConnection(IslandSim& S, uint32_t edgeType, uint32_t handle) {
  preallocateConnections(S, handle);
  if (addConnectionPreallocated(S, edgeType, handle)) S.dirtyEdges[edgeType].push(handle, S.err);
}
SV_HD void addDelayedDirtyEdges(IslandSim& S, uint32_t nbHandles, const uint32_t* handles) {
  while (nbHandles--) {
    const uint32_t h = *handles++;
    S.dirtyEdges[S.edges[h].type].push(h, S.err);
  }
}

SV_HDN void addConnectionToGraph(IslandSim& S, uint32_t handle) {
  const uint32_t instanceHandle = 2 * handle;
  growEdgeInstances(S, instanceHandle + 2 > S.edgeInstances.size ? instanceHandle + 2 : S.edgeInstances.size);
  Edge& edge = S.edges[handle];
  bool activeEdge = false;
  bool kinematicKinematicEdge = true;
  const uint32_t index1 = S.cpu->edgeNodeIndices[instanceHandle].id;
  const uint32_t index2 = S.cpu->edgeNodeIndices[instanceHandle + 1].id;
  auto connectEdge = [&](uint32_t edgeIndex, Node& source) {
    EdgeInstance& instance = S.edgeInstances[edgeIndex];
    instance.next = source.firstEdgeIndex;
    if (source.firstEdgeIndex != INVALID_EDGE) S.edgeInstances[source.firstEdgeIndex].prev = edgeIndex;
    source.firstEdgeIndex = edgeIndex;
    instance.prev = INVALID_EDGE;
  };
  if (index1 != INVALID_NODE) {
    Node& node = S.nodes[index1];
    connectEdge(instanceHandle, node);
    activeEdge = (node.flags & (N_ACTIVE | N_ACTIVATING)) != 0;
    kinematicKinematicEdge = (node.flags & N_KINEMATIC) != 0;
  }
  if (index1 != index2 && index2 != INVALID_NODE) {
    Node& node = S.nodes[index2];
    connectEdge(instanceHandle + 1, node);
    activeEdge |= (node.flags & (N_ACTIVE | N_ACTIVATING)) != 0;
    kinematicKinematicEdge = kinematicKinematicEdge && (node.flags & N_KINEMATIC);
  }
  if (activeEdge && (!kinematicKinematicEdge || edge.type == eCONTACT_MANAGER)) {
    markEdgeActive(S, handle, index1, index2);
    edge.state |= E_ACTIVE;
  }
}

SV_HD void removeConnectionFromGraph(IslandSim& S, uint32_t edgeIndex) {
  const uint32_t index1 = S.cpu->edgeNodeIndices[2 * edgeIndex].id;
  const uint32_t index2 = S.cpu->edgeNodeIndices[2 * edgeIndex + 1].id;
  if (index1 != INVALID_NODE) {
    Node& node = S.nodes[index1];
    if (index2 == S.fastRoute[index1]) S.fastRoute[index1] = INVALID_NODE;
    if (!(node.flags & N_DIRTY)) {
      S.dirtyMap.growAndSet(index1, S.err);
      node.flags |= N_DIRTY;
    }
  }
  if (index2 != INVALID_NODE) {
    Node& node = S.nodes[index2];
    if (index1 == S.fastRoute[index2]) S.fastRoute[index2] = INVALID_NODE;
    if (!(node.flags & N_DIRTY)) {
      S.dirtyMap.growAndSet(index2, S.err);
      node.flags |= N_DIRTY;
    }
  }
}

SV_HD void removeConnection(IslandSim& S, uint32_t edgeIndex) {
  Edge& edge = S.edges[edgeIndex];
  if (!(edge.state & E_PENDING_DESTROYED)) S.destroyedEdges.push(edgeIndex, S.err);
  edge.state |= E_PENDING_DESTROYED;
}

SV_HD void removeConnectionInternal(IslandSim& S, uint32_t edgeIndex) {
  const uint32_t edgeInstanceBase = edgeIndex * 2;
  const uint32_t index1 = S.cpu->edgeNodeIndices[edgeIndex * 2].id;
  const uint32_t index2 = S.cpu->edgeNodeIndices[edgeIndex * 2 + 1].id;
  auto disconnectEdge = [&](uint32_t ei, Node& node) {
    EdgeInstance& instance = S.edgeInstances[ei];
    if (node.firstEdgeIndex == ei)
      node.firstEdgeIndex = instance.next;
    else
      S.edgeInstances[instance.prev].next = instance.next;
    if (instance.next != INVALID_EDGE) S.edgeInstances[instance.next].prev = instance.prev;
    instance.next = INVALID_EDGE;
    instance.prev = INVALID_EDGE;
  };
  if (index1 != INVALID_NODE) disconnectEdge(edgeInstanceBase, S.nodes[index1]);
  if (index2 != INVALID_NODE && index1 != index2) disconnectEdge(edgeInstanceBase + 1, S.nodes[index2]);
}

SV_HDN void activateNode(IslandSim& S, uint32_t index) {
  if (index != INVALID_NODE) {
    Node& node = S.nodes[index];
    if (!(node.flags & (N_ACTIVE | N_ACTIVATING))) {
      if ((node.flags & N_KINEMATIC) && S.activeNodeIndex[index] != INVALID_NODE) {
        const uint32_t activeRefCount = node.activeRefCount;
        node.activeRefCount = 0;
        node.flags &= ~N_ACTIVE;
        markKinematicInactive(S, index);
        node.activeRefCount = activeRefCount;
      }
      node.flags |= N_ACTIVATING;
      S.activeNodeIndex[index] = S.activatingNodes.size;
      S.activatingNodes.push(index, S.err);
    }
    node.flags &= ~N_READY_FOR_SLEEPING;
  }
}

SV_HDN void deactivateNode(IslandSim& S, uint32_t index) {
  if (index != INVALID_NODE) {
    Node& node = S.nodes[index];
    const bool wasActivating = (node.flags & N_ACTIVATING) != 0;
    if (wasActivating) {
      node.flags &= ~N_ACTIVATING;
      const uint32_t replaceIndex = S.activatingNodes[S.activatingNodes.size - 1];
      S.activeNodeIndex[replaceIndex] = S.activeNodeIndex[index];
      S.activatingNodes[S.activeNodeIndex[index]] = replaceIndex;
      S.activatingNodes.size--;
      S.activeNodeIndex[index] = INVALID_NODE;
      if (node.flags & N_KINEMATIC) {
        S.activeNodeIndex[index] = S.activeKinematicNodes.size;
        S.activeKinematicNodes.push(index, S.err);
      }
    }
    node.flags |= N_READY_FOR_SLEEPING;
  }
}
SV_HD void putNodeToSleep(IslandSim& S, uint32_t index) {
  if (index != INVALID_NODE) deactivateNode(S, index);
}

SV_HD void makeEdgeActive(IslandSim& S, uint32_t instIndex, bool testEdgeType) {
  const uint32_t idx = instIndex / 2;
  Edge& edge = S.edges[idx];
  if (!(edge.state & E_ACTIVE) && (!testEdgeType || (edge.type != eCONSTRAINT))) {
    markEdgeActive(S, idx, S.cpu->edgeNodeIndices[idx * 2].id, S.cpu->edgeNodeIndices[idx * 2 + 1].id);
    edge.state |= E_ACTIVE;
  }
}

SV_HDN void activateNodeInternal(IslandSim& S, uint32_t nodeIndex) {
  Node& node = S.nodes[nodeIndex];
  if (!(node.flags & N_ACTIVE)) {
    uint32_t index = node.firstEdgeIndex;
    while (index != INVALID_EDGE) {
      makeEdgeActive(S, index, false);
      index = S.edgeInstances[index].next;
    }
    if (node.flags & N_KINEMATIC)
      markKinematicActive(S, nodeIndex);
    else
      markActive(S, nodeIndex);
    node.flags |= N_ACTIVE;
  }
}

SV_HD void removeEdgeFromActivatingList(IslandSim& S, uint32_t index) {
  Edge& edge = S.edges[index];
  if (edge.state & E_ACTIVATING) {
    for (uint32_t a = 0, count = S.activatedEdges[edge.type].size; a < count; a++) {
      if (S.activatedEdges[edge.type][a] == index) {
        S.activatedEdges[edge.type].replaceWithLast(a);
        break;
      }
    }
    edge.state &= ~E_ACTIVATING;
  }
  const uint32_t index1 = S.cpu->edgeNodeIndices[index * 2].id;
  const uint32_t index2 = S.cpu->edgeNodeIndices[index * 2 + 1].id;
  if (index1 != INVALID_NODE && index2 != INVALID_NODE) {
    S.nodes[index1].activeRefCount--;
    S.nodes[index2].activeRefCount--;
  }
}

SV_HDN void deactivateNodeInternal(IslandSim& S, uint32_t nodeIndex) {
  Node& node = S.nodes[nodeIndex];
  if (node.flags & N_ACTIVE) {
    if (node.flags & N_KINEMATIC)
      markKinematicInactive(S, nodeIndex);
    else
      markInactive(S, nodeIndex);
    node.flags &= ~N_ACTIVE;
    node.flags &= ~N_ACTIVATING;
    uint32_t index = node.firstEdgeIndex;
    while (index != INVALID_EDGE) {
      const EdgeInstance& instance = S.edgeInstances[index];
      const uint32_t outboundNode = S.cpu->edgeNodeIndices[index ^ 1].id;
      if (outboundNode == INVALID_NODE || !(S.nodes[outboundNode].flags & N_ACTIVE)) {
        const uint32_t idx = index / 2;
        Edge& edge = S.edges[idx];
        if (edge.state & E_ACTIVE) {
          edge.state &= ~E_ACTIVE;
          S.activeEdgeCount[edge.type]--;
          removeEdgeFromActivatingList(S, idx);
          S.deactivatingEdges[edge.type].push(idx, S.err);
        }
      }
      index = instance.next;
    }
  }
}

SV_HD void unwindRoute(IslandSim& S, uint32_t traversalIndex, uint32_t lastNode, uint32_t hopCount, uint32_t id) {
  uint32_t currIndex = traversalIndex;
  uint32_t hc = hopCount + 1;
  do {
    TraversalState& state = S.visitedNodes[currIndex];
    S.hopCounts[state.nodeIndex] = hc++;
    S.islandIds[state.nodeIndex] = id;
    S.fastRoute[state.nodeIndex] = lastNode;
    currIndex = state.prevIndex;
    lastNode = state.nodeIndex;
  } while (currIndex != INVALID_NODE);
}

SV_HD void activateIslandInternal(IslandSim& S, const Island& island) {
  uint32_t currentNode = island.rootNode;
  while (currentNode != INVALID_NODE) {
    activateNodeInternal(S, currentNode);
    currentNode = S.nodes[currentNode].nextNode;
  }
}
SV_HD void activateIsland(IslandSim& S, uint32_t islandId) {
  Island& island = S.islands[islandId];
  activateIslandInternal(S, island);
  markIslandActive(S, islandId);
}
SV_HD void deactivateIsland(IslandSim& S, uint32_t islandId) {
  Island& island = S.islands[islandId];
  uint32_t currentNode = island.rootNode;
  while (currentNode != INVALID_NODE) {
    const Node& node = S.nodes[currentNode];
    S.nodesToPutToSleep[node.type].push(currentNode, S.err);
    deactivateNodeInternal(S, currentNode);
    currentNode = node.nextNode;
  }
  markIslandInactive(S, islandId);
}

SV_HDN void wakeIslandsInternal(IslandSim& S, bool flag) {
  const uint32_t originalActiveIslands = S.activeIslands.size;
  if (flag) {
    for (uint32_t a = 0; a < EDGE_TYPES; ++a) {
      for (uint32_t i = 0, count = S.activatedEdges[a].size; i < count; ++i) S.edges[S.activatedEdges[a][i]].state &= ~E_ACTIVATING;
      S.activatedEdges[a].size = 0;
    }
    for (uint32_t a = 0; a < EDGE_TYPES; ++a) S.initialActiveNodeCount[a] = S.activeNodes[a].size;
  }
  for (uint32_t a = 0; a < S.activatingNodes.size; ++a) {
    const uint32_t wakeNode = S.activatingNodes[a];
    const uint32_t islandId = S.islandIds[wakeNode];
    Node& node = S.nodes[wakeNode];
    node.flags &= ~N_ACTIVATING;
    if (islandId != INVALID_ISLAND) {
      if (!S.islandAwake.test(islandId)) markIslandActive(S, islandId);
      S.activeNodeIndex[wakeNode] = INVALID_NODE;
      activateNodeInternal(S, wakeNode);
    } else {
      node.flags |= N_ACTIVE;
      S.activeNodeIndex[wakeNode] = S.activeKinematicNodes.size;
      S.activeKinematicNodes.push(wakeNode, S.err);
      uint32_t index = node.firstEdgeIndex;
      while (index != INVALID_EDGE) {
        const EdgeInstance& edgeInstance = S.edgeInstances[index];
        const uint32_t nodeIndex = S.cpu->edgeNodeIndices[index ^ 1].id;
        if (nodeIndex == INVALID_NODE || S.islandIds[nodeIndex] == INVALID_ISLAND) {
          makeEdgeActive(S, index, true);
        } else {
          const uint32_t connectedIslandId = S.islandIds[nodeIndex];
          if (!S.islandAwake.test(connectedIslandId)) markIslandActive(S, connectedIslandId);
        }
        index = edgeInstance.next;
      }
    }
  }
  S.activatingNodes.size = 0;
  for (uint32_t a = originalActiveIslands; a < S.activeIslands.size; ++a) activateIslandInternal(S, S.islands[S.activeIslands[a]]);
}
SV_HD void wakeIslands(IslandSim& S) { wakeIslandsInternal(S, true); }
SV_HD void wakeIslands2(IslandSim& S) { wakeIslandsInternal(S, false); }

SV_HD void insertNewEdges(IslandSim& S) {
  for (uint32_t i = 0; i < EDGE_TYPES; ++i) {
    for (uint32_t a = 0; a < S.dirtyEdges[i].size; ++a) {
      const uint32_t edgeIndex = S.dirtyEdges[i][a];
      Edge& edge = S.edges[edgeIndex];
      if (!(edge.state & E_PENDING_DESTROYED)) {
        if (!(edge.state & E_INSERTED)) {
          addConnectionToGraph(S, edgeIndex);
          edge.state |= E_INSERTED;
        }
      }
    }
  }
}

SV_HDN void removeDestroyedEdges(IslandSim& S) {
  for (uint32_t a = 0; a < S.destroyedEdges.size; ++a) {
    const uint32_t edgeIndex = S.destroyedEdges[a];
    const Edge& edge = S.edges[edgeIndex];
    if (edge.state & E_PENDING_DESTROYED) {
      if (!(edge.state & E_IN_DIRTY_LIST) && (edge.state & E_INSERTED)) {
        removeConnectionInternal(S, edgeIndex);
        removeConnectionFromGraph(S, edgeIndex);
      }
    }
  }
}

SV_HD uint32_t addNodeToIsland(IslandSim& S, uint32_t nodeIndex1, uint32_t nodeIndex2, uint32_t islandId2, bool active1, bool active2) {
  if (nodeIndex1 != INVALID_NODE) {
    if (!(S.nodes[nodeIndex1].flags & N_KINEMATIC)) {
      Island& island = S.islands[islandId2];
      Node& lastNode = S.nodes[island.lastNode];
      Node& node = S.nodes[nodeIndex1];
      lastNode.nextNode = nodeIndex1;
      node.prevNode = island.lastNode;
      island.lastNode = nodeIndex1;
      island.nodeCount[node.type]++;
      S.islandIds[nodeIndex1] = islandId2;
      S.hopCounts[nodeIndex1] = S.hopCounts[nodeIndex2] + 1;
      S.fastRoute[nodeIndex1] = nodeIndex2;
      if (active1 || active2) {
        if (!S.islandAwake.test(islandId2)) activateIsland(S, islandId2);
        if (!active1) activateNodeInternal(S, nodeIndex1);
      }
    } else if (active1 && !active2) {
      activateIsland(S, islandId2);
    }
  } else {
    Node& node = S.nodes[nodeIndex2];
    node.staticTouchCount++;
    S.islandStaticTouchCount[islandId2]++;
  }
  return islandId2;
}

SV_HDN void mergeIslandsInternal(IslandSim& S, Island& island0, Island& island1, uint32_t islandId0, uint32_t islandId1, uint32_t nodeIndex0,
                                 uint32_t nodeIndex1) {
  const uint32_t extraPath = S.hopCounts[nodeIndex0] + S.hopCounts[nodeIndex1] + 1;
  uint32_t islandNode = island1.rootNode;
  while (islandNode != INVALID_NODE) {
    S.hopCounts[islandNode] += extraPath;
    S.islandIds[islandNode] = islandId0;
    islandNode = S.nodes[islandNode].nextNode;
  }
  S.hopCounts[nodeIndex1] = S.hopCounts[nodeIndex0] + 1;
  Node& lastNode = S.nodes[island0.lastNode];
  Node& firstNode = S.nodes[island1.rootNode];
  lastNode.nextNode = island1.rootNode;
  firstNode.prevNode = island0.lastNode;
  island0.lastNode = island1.lastNode;
  S.islandStaticTouchCount[islandId0] += S.islandStaticTouchCount[islandId1];
  for (uint32_t a = 0; a < EDGE_TYPES; ++a) {
    if (island0.lastEdge[a] != INVALID_EDGE)
      S.edges[island0.lastEdge[a]].nextIslandEdge = island1.firstEdge[a];
    else
      island0.firstEdge[a] = island1.firstEdge[a];
    if (island1.firstEdge[a] != INVALID_EDGE) {
      S.edges[island1.firstEdge[a]].prevIslandEdge = island0.lastEdge[a];
      island0.lastEdge[a] = island1.lastEdge[a];
    }
    island0.edgeCount[a] += island1.edgeCount[a];
    island1.firstEdge[a] = INVALID_EDGE;
    island1.lastEdge[a] = INVALID_EDGE;
    island1.edgeCount[a] = 0;
  }
  for (uint32_t a = 0; a < NODE_TYPES; ++a) {
    island0.nodeCount[a] += island1.nodeCount[a];
    island1.nodeCount[a] = 0;
  }
  island1.lastNode = INVALID_NODE;
  island1.rootNode = INVALID_NODE;
  S.islandStaticTouchCount[islandId1] = 0;
  if (island1.activeIndex != INVALID_ISLAND) markIslandInactive(S, islandId1);
}

SV_HD uint32_t mergeIslands(IslandSim& S, uint32_t island0, uint32_t island1, uint32_t node0, uint32_t node1) {
  Island& is0 = S.islands[island0];
  Island& is1 = S.islands[island1];
  uint32_t totalSize0 = 0, totalSize1 = 0;
  for (uint32_t i = 0; i < NODE_TYPES; ++i) {
    totalSize0 += is0.nodeCount[i];
    totalSize1 += is1.nodeCount[i];
  }
  if (totalSize0 > totalSize1) {
    mergeIslandsInternal(S, is0, is1, island0, island1, node0, node1);
    S.islandAwake.reset(island1);
    S.islandHandles.freeHandle(island1, S.err);
    S.fastRoute[node1] = node0;
    return island0;
  } else {
    mergeIslandsInternal(S, is1, is0, island1, island0, node1, node0);
    S.islandAwake.reset(island0);
    S.islandHandles.freeHandle(island0, S.err);
    S.fastRoute[node0] = node1;
    return island1;
  }
}

SV_HDN void processNewEdges(IslandSim& S) {
  insertNewEdges(S);
  // mHopCounts / mFastRoute resize(mNodes.size()) — 우리 배열은 노드와 같은 크기
  for (uint32_t i = 0; i < EDGE_TYPES; ++i) {
    for (uint32_t a = 0; a < S.dirtyEdges[i].size; ++a) {
      const uint32_t edgeIndex = S.dirtyEdges[i][a];
      const Edge& edge = S.edges[edgeIndex];
      if (!(edge.state & E_PENDING_DESTROYED)) {
        const uint32_t index1 = S.cpu->edgeNodeIndices[2 * edgeIndex].id;
        const uint32_t index2 = S.cpu->edgeNodeIndices[2 * edgeIndex + 1].id;
        const uint32_t islandId1 = index1 == INVALID_NODE ? INVALID_ISLAND : S.islandIds[index1];
        const uint32_t islandId2 = index2 == INVALID_NODE ? INVALID_ISLAND : S.islandIds[index2];
        const bool active1 = index1 != INVALID_NODE && (S.nodes[index1].flags & N_ACTIVE);
        const bool active2 = index2 != INVALID_NODE && (S.nodes[index2].flags & N_ACTIVE);
        uint32_t islandId = INVALID_ISLAND;
        if (islandId1 == INVALID_ISLAND && islandId2 == INVALID_ISLAND) {
        } else if (islandId1 == islandId2) {
          islandId = islandId1;
          const uint32_t hopCount1 = S.hopCounts[index1];
          const uint32_t hopCount2 = S.hopCounts[index2];
          if ((hopCount1 + 1) < hopCount2) {
            S.hopCounts[index2] = hopCount1 + 1;
            S.fastRoute[index2] = index1;
          } else if ((hopCount2 + 1) < hopCount1) {
            S.hopCounts[index1] = hopCount2 + 1;
            S.fastRoute[index1] = index2;
          }
        } else if (islandId1 == INVALID_ISLAND) {
          islandId = addNodeToIsland(S, index1, index2, islandId2, active1, active2);
        } else if (islandId2 == INVALID_ISLAND) {
          islandId = addNodeToIsland(S, index2, index1, islandId1, active2, active1);
        } else {
          if (active1 || active2) {
            if (!S.islandAwake.test(islandId1)) activateIsland(S, islandId1);
            if (!S.islandAwake.test(islandId2)) activateIsland(S, islandId2);
          }
          islandId = mergeIslands(S, islandId1, islandId2, index1, index2);
        }
        if (islandId != INVALID_ISLAND) addEdgeToIsland(S, S.islands[islandId], edgeIndex);
      }
    }
  }
}

SV_HD bool tryFastPath(IslandSim& S, uint32_t startNode, uint32_t targetNode, uint32_t islandId) {
  uint32_t currentNode = startNode;
  const uint32_t currentVisitedNodes = S.visitedNodes.size;
  uint32_t depth = 0;
  bool found = false;
  do {
    if (S.visitedState.test(currentNode)) {
      found = S.islandIds[currentNode] != INVALID_ISLAND;
      break;
    }
    if (currentNode == targetNode) {
      found = true;
      break;
    }
    S.visitedNodes.push(TraversalState{currentNode, S.visitedNodes.size, S.visitedNodes.size - 1, depth++}, S.err);
    S.islandIds[currentNode] = INVALID_ISLAND;
    S.visitedState.set(currentNode);
    currentNode = S.fastRoute[currentNode];
  } while (currentNode != INVALID_NODE);
  for (uint32_t a = currentVisitedNodes; a < S.visitedNodes.size; ++a) S.islandIds[S.visitedNodes[a].nodeIndex] = islandId;
  if (!found) {
    for (uint32_t a = currentVisitedNodes; a < S.visitedNodes.size; ++a) S.visitedState.reset(S.visitedNodes[a].nodeIndex);
    S.visitedNodes.size = currentVisitedNodes;
  }
  return found;
}

SV_HDN bool findRoute(IslandSim& S, uint32_t startNode, uint32_t targetNode, uint32_t islandId) {
  if (S.fastRoute[startNode] != INVALID_NODE) {
    if (tryFastPath(S, startNode, targetNode, islandId)) return true;
  }
  S.islandIds[startNode] = INVALID_ISLAND;
  const uint32_t startTraversal = S.visitedNodes.size;
  S.visitedNodes.push(TraversalState{startNode, S.visitedNodes.size, INVALID_NODE, 0}, S.err);
  S.visitedState.set(startNode);
  pqPush(S, QueueElement{startTraversal, S.hopCounts[startNode]});
  do {
    const QueueElement currentQE = pqPop(S);
    const TraversalState currentState = S.visitedNodes[currentQE.state];
    const Node& currentNode = S.nodes[currentState.nodeIndex];
    uint32_t edge = currentNode.firstEdgeIndex;
    while (edge != INVALID_EDGE) {
      const EdgeInstance& instance = S.edgeInstances[edge];
      const uint32_t nextIndex = S.cpu->edgeNodeIndices[edge ^ 1].id;
      if (nextIndex != INVALID_NODE && !(S.nodes[nextIndex].flags & N_KINEMATIC)) {
        if (nextIndex == targetNode) {
          unwindRoute(S, currentState.currentIndex, nextIndex, 0, islandId);
          return true;
        }
        if (S.visitedState.test(nextIndex)) {
          const uint32_t visitedIslandId = S.islandIds[nextIndex];
          if (visitedIslandId != INVALID_ISLAND) {
            unwindRoute(S, currentState.currentIndex, nextIndex, S.hopCounts[nextIndex], islandId);
            return true;
          }
        } else {
          const uint32_t st = S.visitedNodes.size;
          S.visitedNodes.push(TraversalState{nextIndex, S.visitedNodes.size, currentState.currentIndex, currentState.depth + 1}, S.err);
          pqPush(S, QueueElement{st, S.hopCounts[nextIndex]});
          S.visitedState.set(nextIndex);
          S.islandIds[nextIndex] = INVALID_ISLAND;
        }
      }
      edge = instance.next;
    }
  } while (S.pq.size);
  return false;
}

// destroyedNodes: SimpleIslandManager 의 mDestroyedNodes
SV_HDN void processLostEdges(IslandSim& S, const uint32_t* destroyedNodes, uint32_t nbDestroyedNodes, bool allowDeactivation,
                             bool permitKinematicDeactivation) {
  S.visitedState.resizeAndClear(S.nodes.size, S.err);
  S.pq.size = 0;
  {
    for (uint32_t a = 0; a < S.destroyedEdges.size; ++a) {
      const uint32_t lostIndex = S.destroyedEdges[a];
      Edge& lostEdge = S.edges[lostIndex];
      if ((lostEdge.state & E_PENDING_DESTROYED) && !(lostEdge.state & E_IN_DIRTY_LIST)) {
        if (!(lostEdge.state & E_REPORT_ONLY_DESTROY) && (lostEdge.state & E_INSERTED)) {
          const uint32_t index1 = S.cpu->edgeNodeIndices[lostIndex * 2].id;
          const uint32_t index2 = S.cpu->edgeNodeIndices[lostIndex * 2 + 1].id;
          uint32_t islandId = INVALID_ISLAND;
          if (index1 != INVALID_NODE && index2 != INVALID_NODE) {
            islandId = S.islandIds[index1] != INVALID_ISLAND ? S.islandIds[index1] : S.islandIds[index2];
          } else if (index1 != INVALID_NODE) {
            Node& node = S.nodes[index1];
            if (!(node.flags & N_KINEMATIC)) {
              islandId = S.islandIds[index1];
              node.staticTouchCount--;
              S.islandStaticTouchCount[islandId]--;
            }
          } else if (index2 != INVALID_NODE) {
            Node& node = S.nodes[index2];
            if (!(node.flags & N_KINEMATIC)) {
              islandId = S.islandIds[index2];
              node.staticTouchCount--;
              S.islandStaticTouchCount[islandId]--;
            }
          }
          if (islandId != INVALID_ISLAND) removeEdgeFromIsland(S, S.islands[islandId], lostIndex);
        }
        lostEdge.state &= ~E_INSERTED;
      }
    }
  }
  if (allowDeactivation) {
    // PxBitMap::Iterator: 오름차순, 현재 낱말은 읽어 둔 값으로 (루프 안에서 dirtyMap 을 바꾸지 않음)
    for (uint32_t wi = 0; wi < S.dirtyMap.words; ++wi) {
      uint32_t block = S.dirtyMap.w[wi];
      while (block) {
        uint32_t low = 0;
        while (!((block >> low) & 1u)) ++low;
        block &= block - 1;
        const uint32_t dirtyIdx = (wi << 5) | low;
        S.pq.size = 0;
        S.visitedNodes.size = 0;
        const uint32_t dirtyNodeIndex = dirtyIdx;
        Node& dirtyNode = S.nodes[dirtyNodeIndex];
        if (!(dirtyNode.flags & N_KINEMATIC) && !(dirtyNode.flags & N_DELETED) && !S.visitedState.test(dirtyNodeIndex)) {
          const uint32_t islandId = S.islandIds[dirtyNodeIndex];
          const Island& findIsland = S.islands[islandId];
          const uint32_t searchNode = findIsland.rootNode;
          if (searchNode != dirtyNodeIndex) {
            if (findRoute(S, dirtyNodeIndex, searchNode, islandId)) {
              for (uint32_t b = 0; b < S.visitedNodes.size; ++b) {
                TraversalState& state = S.visitedNodes[b];
                if (S.islandIds[state.nodeIndex] == INVALID_ISLAND) {
                  S.hopCounts[state.nodeIndex] = S.hopCounts[S.visitedNodes[state.prevIndex].nodeIndex] + 1;
                  S.fastRoute[state.nodeIndex] = S.visitedNodes[state.prevIndex].nodeIndex;
                  S.islandIds[state.nodeIndex] = islandId;
                }
              }
            } else {
              Island& oldIsland = S.islands[islandId];
              uint32_t totalStaticTouchCount = 0;
              uint32_t nodeCount[NODE_TYPES];
              for (uint32_t t = 0; t < NODE_TYPES; ++t) nodeCount[t] = 0;
              for (uint32_t t = 0; t < EDGE_TYPES; ++t) S.islandSplitEdges[t].size = 0;
              for (uint32_t a = 0; a < S.visitedNodes.size; ++a) {
                const uint32_t index = S.visitedNodes[a].nodeIndex;
                Node& node = S.nodes[index];
                if (node.nextNode != INVALID_NODE)
                  S.nodes[node.nextNode].prevNode = node.prevNode;
                else
                  oldIsland.lastNode = node.prevNode;
                if (node.prevNode != INVALID_NODE) S.nodes[node.prevNode].nextNode = node.nextNode;
                nodeCount[node.type]++;
                node.nextNode = INVALID_NODE;
                node.prevNode = INVALID_NODE;
                totalStaticTouchCount += node.staticTouchCount;
                uint32_t idx = node.firstEdgeIndex;
                while (idx != INVALID_EDGE) {
                  const EdgeInstance& instance = S.edgeInstances[idx];
                  const uint32_t edgeIndex = idx / 2;
                  const Edge& edge = S.edges[edgeIndex];
                  if (!(idx & 1) || (S.cpu->edgeNodeIndices[idx & (~1u)].id == INVALID_NODE ||
                                     (S.nodes[S.cpu->edgeNodeIndices[idx & (~1u)].id].flags & N_KINEMATIC))) {
                    S.islandSplitEdges[edge.type].push(edgeIndex, S.err);
                    removeEdgeFromIsland(S, oldIsland, edgeIndex);
                  }
                  idx = instance.next;
                }
              }
              S.islandStaticTouchCount[islandId] -= totalStaticTouchCount;
              for (uint32_t i = 0; i < NODE_TYPES; ++i) oldIsland.nodeCount[i] -= nodeCount[i];
              const uint32_t newIslandHandle = S.islandHandles.getHandle();
              growIslands(S, newIslandHandle + 1 > S.islands.size ? newIslandHandle + 1 : S.islands.size);
              Island& newIsland = S.islands[newIslandHandle];
              if (S.islandAwake.test(islandId)) {
                newIsland.activeIndex = S.activeIslands.size;
                S.activeIslands.push(newIslandHandle, S.err);
                S.islandAwake.growAndSet(newIslandHandle, S.err);
              } else {
                S.islandAwake.growAndReset(newIslandHandle, S.err);
              }
              newIsland.rootNode = dirtyNodeIndex;
              S.hopCounts[dirtyNodeIndex] = 0;
              S.islandIds[dirtyNodeIndex] = newIslandHandle;
              S.nodes[dirtyNodeIndex].prevNode = INVALID_NODE;
              S.fastRoute[dirtyNodeIndex] = INVALID_NODE;
              for (uint32_t i = 0; i < NODE_TYPES; ++i) nodeCount[i] = 0;
              nodeCount[dirtyNode.type] = 1;
              for (uint32_t a = 1; a < S.visitedNodes.size; ++a) {
                const uint32_t index = S.visitedNodes[a].nodeIndex;
                Node& thisNode = S.nodes[index];
                const uint32_t prevNodeIndex = S.visitedNodes[a - 1].nodeIndex;
                thisNode.prevNode = prevNodeIndex;
                S.nodes[prevNodeIndex].nextNode = index;
                nodeCount[thisNode.type]++;
                S.islandIds[index] = newIslandHandle;
                S.hopCounts[index] = S.visitedNodes[a].depth;
                S.fastRoute[index] = S.visitedNodes[S.visitedNodes[a].prevIndex].nodeIndex;
              }
              for (uint32_t i = 0; i < NODE_TYPES; ++i) newIsland.nodeCount[i] = nodeCount[i];
              const uint32_t lastIndex = S.visitedNodes[S.visitedNodes.size - 1].nodeIndex;
              S.nodes[lastIndex].nextNode = INVALID_NODE;
              newIsland.lastNode = lastIndex;
              S.islandStaticTouchCount[newIslandHandle] = totalStaticTouchCount;
              for (uint32_t j = 0; j < EDGE_TYPES; ++j) {
                Arr<uint32_t>& splitEdges = S.islandSplitEdges[j];
                const uint32_t splitEdgeSize = splitEdges.size;
                if (splitEdgeSize) {
                  splitEdges.push(INVALID_EDGE, S.err);
                  S.edges[splitEdges[0]].nextIslandEdge = splitEdges[1];
                  for (uint32_t a = 1; a < splitEdgeSize; ++a) {
                    const uint32_t edgeIndex = splitEdges[a];
                    Edge& edge = S.edges[edgeIndex];
                    edge.nextIslandEdge = splitEdges[a + 1];
                    edge.prevIslandEdge = splitEdges[a - 1];
                  }
                  newIsland.firstEdge[j] = splitEdges[0];
                  newIsland.lastEdge[j] = splitEdges[splitEdgeSize - 1];
                  newIsland.edgeCount[j] = splitEdgeSize;
                }
              }
            }
          }
        }
        dirtyNode.flags &= ~N_DIRTY;
      }
    }
    S.dirtyMap.clearAll();
  }
  {
    for (uint32_t a = 0; a < S.destroyedEdges.size; ++a) {
      const uint32_t index = S.destroyedEdges[a];
      Edge& edge = S.edges[index];
      if (edge.state & E_PENDING_DESTROYED) {
        if (edge.state & E_ACTIVE) {
          removeEdgeFromActivatingList(S, index);
          S.activeEdgeCount[edge.type]--;
        }
        edge = edgeDefault();
      }
    }
    S.destroyedEdges.size = 0;
  }
  {
    for (uint32_t a = 0; a < nbDestroyedNodes; ++a) {
      const uint32_t nodeIndex = destroyedNodes[a];
      const uint32_t islandId = S.islandIds[nodeIndex];
      Node& node = S.nodes[nodeIndex];
      if (islandId != INVALID_ISLAND) {
        Island& island = S.islands[islandId];
        removeNodeFromIsland(S, island, nodeIndex);
        S.islandIds[nodeIndex] = INVALID_ISLAND;
        uint32_t nodeCountTotal = 0;
        for (uint32_t t = 0; t < NODE_TYPES; ++t) nodeCountTotal += island.nodeCount[t];
        if (nodeCountTotal == 0) {
          S.islandHandles.freeHandle(islandId, S.err);
          if (island.activeIndex != INVALID_ISLAND) {
            const uint32_t replaceId = S.activeIslands[S.activeIslands.size - 1];
            Island& replaceIsland = S.islands[replaceId];
            replaceIsland.activeIndex = island.activeIndex;
            S.activeIslands[island.activeIndex] = replaceId;
            S.activeIslands.size--;
            island.activeIndex = INVALID_ISLAND;
            S.islandStaticTouchCount[islandId] -= node.staticTouchCount;
          }
          S.islandAwake.reset(islandId);
          island.lastNode = INVALID_NODE;
          island.rootNode = INVALID_NODE;
          island.activeIndex = INVALID_ISLAND;
        }
      }
      if (node.flags & N_KINEMATIC) {
        if (S.activeNodeIndex[nodeIndex] != INVALID_NODE) markKinematicInactive(S, nodeIndex);
      } else {
        if (S.activeNodeIndex[nodeIndex] != INVALID_NODE) markInactive(S, nodeIndex);
      }
      node.flags |= N_DELETED;
    }
  }
  if (allowDeactivation) {
    for (uint32_t a = 0; a < S.activeIslands.size; a++) S.islandAwake.reset(S.activeIslands[a]);
    for (uint32_t a = S.activeKinematicNodes.size; a > 0; --a) {
      const uint32_t kinematicIndex = S.activeKinematicNodes[a - 1];
      Node& kinematicNode = S.nodes[kinematicIndex];
      if (kinematicNode.flags & N_READY_FOR_SLEEPING) {
        if (permitKinematicDeactivation) {
          kinematicNode.flags &= ~N_ACTIVE;
          markKinematicInactive(S, kinematicIndex);
        }
      } else {
        uint32_t edgeId = kinematicNode.firstEdgeIndex;
        while (edgeId != INVALID_EDGE) {
          const EdgeInstance& instance = S.edgeInstances[edgeId];
          const uint32_t outNode = S.cpu->edgeNodeIndices[edgeId ^ 1].id;
          if (outNode != INVALID_NODE) {
            const uint32_t islandId = S.islandIds[outNode];
            if (islandId != INVALID_ISLAND) S.islandAwake.set(islandId);
          }
          edgeId = instance.next;
        }
      }
    }
    for (uint32_t a = S.activeIslands.size; a > 0; --a) {
      const uint32_t islandId = S.activeIslands[a - 1];
      const Island& island = S.islands[islandId];
      bool canDeactivate = !S.islandAwake.test(islandId);
      S.islandAwake.set(islandId);
      if (canDeactivate) {
        uint32_t nodeId = island.rootNode;
        while (nodeId != INVALID_NODE) {
          Node& node = S.nodes[nodeId];
          if (!(node.flags & N_READY_FOR_SLEEPING)) {
            canDeactivate = false;
            break;
          }
          nodeId = node.nextNode;
        }
        if (canDeactivate) deactivateIsland(S, islandId);
      }
    }
  }
  {
    for (uint32_t i = 0; i < EDGE_TYPES; ++i) {
      for (uint32_t a = 0; a < S.dirtyEdges[i].size; ++a) S.edges[S.dirtyEdges[i][a]].state &= ~E_IN_DIRTY_LIST;
      S.dirtyEdges[i].size = 0;
    }
  }
}

SV_HD void clearDeactivations(IslandSim& S) {
  for (uint32_t i = 0; i < NODE_TYPES; ++i) {
    S.nodesToPutToSleep[i].size = 0;
    S.deactivatingEdges[i].size = 0;
  }
}

// ---------------- SimpleIslandManager (PxsSimpleIslandManager.cpp)
struct IslandManager {
  HandleManager nodeHandles, edgeHandles;
  Arr<uint32_t> destroyedNodes;  // 노드 번호
  Arr<uint32_t> destroyedEdges;
  CpuData cpu;
  uint32_t* constraintOrCm;  // 간선 -> 접촉 관리자/제약 번호 (NONE = 없음)
  Bits connectedMap;
  IslandSim accurate, speculative;
  uint32_t err;
};

SV_HD void ensureEdgeTables(IslandManager& M, uint32_t handle) {
  if (2 * handle + 1 >= M.cpu.cap) M.err |= 128u;
}

SV_HD uint32_t addNode(IslandManager& M, bool isActive, bool isKinematic, uint8_t type, uint32_t object) {
  const uint32_t handle = M.nodeHandles.getHandle();
  addNode(M.accurate, isActive, isKinematic, type, handle, object);
  addNode(M.speculative, isActive, isKinematic, type, handle, object);
  return handle;
}
SV_HD void removeNode(IslandManager& M, uint32_t index) { M.destroyedNodes.push(index, M.err); }

SV_HD uint32_t addEdge(IslandManager& M, uint32_t edgeObj, NodeIndex n1, NodeIndex n2) {
  const uint32_t handle = M.edgeHandles.getHandle();
  ensureEdgeTables(M, handle);
  M.cpu.edgeNodeIndices[2 * handle] = n1;
  M.cpu.edgeNodeIndices[2 * handle + 1] = n2;
  M.constraintOrCm[handle] = edgeObj;
  return handle;
}
SV_HD uint32_t resizeEdgeArrays(IslandManager& M, uint32_t handle, bool flag) {
  M.connectedMap.ensure(handle, M.err);
  if (flag)
    M.connectedMap.reset(handle);
  else
    M.connectedMap.set(handle);
  return handle;
}
// preallocateContactManagers + addPreallocatedContactManager (Sc::Scene::islandInsertion 의 다중 스레드판을 한 번에)
SV_HD uint32_t addContactManager(IslandManager& M, uint32_t cm, NodeIndex n1, NodeIndex n2, uint32_t edgeType) {
  const uint32_t handle = addEdge(M, cm, n1, n2);
  addConnection(M.speculative, edgeType, handle);
  return resizeEdgeArrays(M, handle, true);
}
// Sc::Scene::islandInsertion 의 다중 스레드판 (PxsSimpleIslandManager.cpp:155-219): 번호를 먼저 다 받고, 쌍마다 채우고,
// 추측 섬 시뮬의 dirty 목록은 addDelayedDirtyEdges 로 한꺼번에 (Sc 가 부름)
SV_HD void preallocateContactManagers(IslandManager& M, uint32_t nb, uint32_t* handles) {
  uint32_t maxHandle = 0;
  for (uint32_t i = 0; i < nb; i++) {
    const uint32_t handle = M.edgeHandles.getHandle();
    handles[i] = handle;
    if (handle > maxHandle) maxHandle = handle;
  }
  ensureEdgeTables(M, maxHandle);
  preallocateConnections(M.speculative, maxHandle);
  M.connectedMap.ensure(maxHandle, M.err);
}
SV_HD bool addPreallocatedContactManager(IslandManager& M, uint32_t handle, uint32_t cm, NodeIndex n1, NodeIndex n2, uint32_t edgeType) {
  M.cpu.edgeNodeIndices[2 * handle] = n1;
  M.cpu.edgeNodeIndices[2 * handle + 1] = n2;
  M.constraintOrCm[handle] = cm;
  const bool status = addConnectionPreallocated(M.speculative, edgeType, handle);
  M.connectedMap.reset(handle);
  return status;
}
SV_HD uint32_t addConstraint(IslandManager& M, uint32_t constraint, NodeIndex n1, NodeIndex n2) {
  const uint32_t handle = addEdge(M, constraint, n1, n2);
  addConnection(M.accurate, eCONSTRAINT, handle);
  addConnection(M.speculative, eCONSTRAINT, handle);
  return resizeEdgeArrays(M, handle, false);
}
SV_HD void activateNode(IslandManager& M, uint32_t i) {
  activateNode(M.accurate, i);
  activateNode(M.speculative, i);
}
SV_HD void deactivateNode(IslandManager& M, uint32_t i) {
  deactivateNode(M.accurate, i);
  deactivateNode(M.speculative, i);
}
SV_HD void putNodeToSleep(IslandManager& M, uint32_t i) {
  putNodeToSleep(M.accurate, i);
  putNodeToSleep(M.speculative, i);
}
SV_HD void removeConnection(IslandManager& M, uint32_t edgeIndex) {
  if (edgeIndex == INVALID_EDGE) return;
  M.destroyedEdges.push(edgeIndex, M.err);
  removeConnection(M.speculative, edgeIndex);
  if (M.connectedMap.test(edgeIndex)) {
    removeConnection(M.accurate, edgeIndex);
    M.connectedMap.reset(edgeIndex);
  }
  M.constraintOrCm[edgeIndex] = INVALID_EDGE;
}
SV_HD void firstPassIslandGen(IslandManager& M) {
  clearDeactivations(M.speculative);
  wakeIslands(M.speculative);
  processNewEdges(M.speculative);
  removeDestroyedEdges(M.speculative);
  processLostEdges(M.speculative, M.destroyedNodes.d, M.destroyedNodes.size, false, false);
}
SV_HD void additionalSpeculativeActivation(IslandManager& M) { wakeIslands2(M.speculative); }
SV_HD void secondPassIslandGenPart1(IslandManager& M) {
  wakeIslands(M.accurate);
  processNewEdges(M.accurate);
}
SV_HD void secondPassIslandGenPart2(IslandManager& M) {
  removeDestroyedEdges(M.accurate);
  processLostEdges(M.accurate, M.destroyedNodes.d, M.destroyedNodes.size, false, false);
  for (uint32_t a = 0; a < M.destroyedNodes.size; ++a) M.nodeHandles.freeHandle(M.destroyedNodes[a], M.err);
  M.destroyedNodes.size = 0;
}
// thirdPassIslandGen: 두 섬 시뮬의 3차 작업(서로 독립) 다음 PostThirdPassTask
SV_HD void thirdPassIslandGen(IslandManager& M) {
  clearDeactivations(M.accurate);
  removeDestroyedEdges(M.speculative);
  processLostEdges(M.speculative, M.destroyedNodes.d, M.destroyedNodes.size, true, true);
  removeDestroyedEdges(M.accurate);
  processLostEdges(M.accurate, M.destroyedNodes.d, M.destroyedNodes.size, true, true);
  for (uint32_t a = 0; a < M.destroyedNodes.size; ++a) M.nodeHandles.freeHandle(M.destroyedNodes[a], M.err);
  M.destroyedNodes.size = 0;
  for (uint32_t a = 0; a < M.destroyedEdges.size; ++a) M.edgeHandles.freeHandle(M.destroyedEdges[a], M.err);
  M.destroyedEdges.size = 0;
}
SV_HD void setEdgeConnected(IslandManager& M, uint32_t edgeIndex, uint32_t edgeType) {
  if (!M.connectedMap.test(edgeIndex)) {
    addConnection(M.accurate, edgeType, edgeIndex);
    M.connectedMap.set(edgeIndex);
  }
}
SV_HD void setEdgeDisconnected(IslandManager& M, uint32_t edgeIndex) {
  if (M.connectedMap.test(edgeIndex)) {
    removeConnection(M.accurate, edgeIndex);
    M.connectedMap.reset(edgeIndex);
  }
}

}  // namespace ig
}  // namespace eng
