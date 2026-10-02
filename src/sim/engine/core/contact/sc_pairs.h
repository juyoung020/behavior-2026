// contact "쌍 관리층" — PhysX Sc 층의 접촉 쌍 부분을 손으로 옮긴 것 (층 1, 호스트, PhysX 링크 없음).
//
// 하는 일 (PhysX 5.6.1 원본, physx/source/ 기준):
//  - 넓은 단계가 낸 새 겹침 거르기(filter): simulationcontroller/src/ScFiltering.cpp:356-607 (runOverlapFilters, filterRbCollisionPairShared,
//    filterRbCollisionPairSecondStage, checkRbPairFlags, filterKinematics, filterJointedBodies, filterArticulationLinks)
//  - 상호작용 만들기·없애기: ScNPhaseCore.cpp:89-302,946-1082 (createRbElementInteraction, shouldSwapBodies, releaseElementPair, lostTouchReports),
//    ScShapeInteraction.cpp:38-148,661-1141 (생성자·소멸자, updateFlags, updateState, onActivate/onDeactivate, createManager),
//    ScShapeInteraction.h:257-308 (updateManager, destroyManager, activeManagerAllowed), ScActorSim.cpp:101-120 (행위자 상호작용 목록)
//  - Sc 한 스텝의 쌍 처리 순서: ScPipeline.cpp:477-1200 (finishBroadPhase, preallocateContactManagers, postBroadPhaseStage2, registerContactManagers,
//    registerInteractions), :2147-2440 (processLostContacts 1·2·3, destroyManagers, processNarrowPhaseLostTouchEvents)
//  - 접촉 관리자 풀 번호: common/src/CmPool.h:48-244 (PoolList: preallocate·get·put 의 빈 칸 목록 순서)
//  - 좁은 단계 목록: lowlevel/software/src/PxsNphaseImplementationContext.cpp:643-1009 (register/unregister/refresh, appendContactManagers,
//    unregisterContactManagerInternal 바꿔치기 삭제), 닿음 변화 비트맵과 사건 순서: :321-396 (processCms), PxsContext.cpp:528-604 (fillManagerTouchEvents)
//
// 섬 관리(IG::SimpleIslandManager)는 solver 소유다. 이 층은 PhysX 가 섬 관리자를 부르는 자리에서 IslandHooks 를 같은 순서·같은 인자로 부른다
// (문서 12.3 "contact 쌍 관리층 ↔ solver 섬"). 깨우기(internalWakeUp)·잃은 닿음 목록(addToLostTouchList)도 갈고리로 넘긴다.
//
// 번호 규약: 모양 = ElementSim 번호(넓은 단계 경계 번호와 같음), 행위자 = 엔진 행위자 번호, 상호작용·접촉 관리자는 이 층이 매긴다.
// 접촉 관리자 번호 = PhysX PxsContactManager::getIndex() 와 같은 값(풀 번호). 상호작용 번호는 PhysX 에 없는 내부 번호(순서 영향 없음).
//
// 아직 안 한 것(보고): 접촉 보고(ActorPairReport·보고 흐름), CCD, 트리거 겹침 처리(트리거 상호작용은 만들고 없애기만), 사용자 재거르기(API),
// 입자·변형체. 이것들은 BEHAVIOR 동역학 비트에 영향이 없거나 장면에 없다.
#pragma once
#include <cstdint>
#include <cstring>

#include "core/contact/sc_containers.h"

namespace eng {
namespace contact {
namespace sc {

// ---- PhysX 상수 (include/PxFiltering.h, include/PxActor.h, source/simulationcontroller/src/*.h, lowlevel 헤더)
namespace PairFlag {
enum : uint32_t {
  eSOLVE_CONTACT = 1u << 0, eMODIFY_CONTACTS = 1u << 1, eNOTIFY_TOUCH_FOUND = 1u << 2, eNOTIFY_TOUCH_PERSISTS = 1u << 3,
  eNOTIFY_TOUCH_LOST = 1u << 4, eNOTIFY_TOUCH_CCD = 1u << 5, eNOTIFY_THRESHOLD_FORCE_FOUND = 1u << 6, eNOTIFY_THRESHOLD_FORCE_PERSISTS = 1u << 7,
  eNOTIFY_THRESHOLD_FORCE_LOST = 1u << 8, eNOTIFY_CONTACT_POINTS = 1u << 9, eDETECT_DISCRETE_CONTACT = 1u << 10, eDETECT_CCD_CONTACT = 1u << 11,
  ePRE_SOLVER_VELOCITY = 1u << 12, ePOST_SOLVER_VELOCITY = 1u << 13, eCONTACT_EVENT_POSE = 1u << 14, eNEXT_FREE = 1u << 15,
  eCONTACT_DEFAULT = eSOLVE_CONTACT | eDETECT_DISCRETE_CONTACT,
  eTRIGGER_DEFAULT = eNOTIFY_TOUCH_FOUND | eNOTIFY_TOUCH_LOST | eDETECT_DISCRETE_CONTACT
};
}
namespace FilterFlag {
enum : uint32_t { eKILL = 1u << 0, eSUPPRESS = 1u << 1, eCALLBACK = 1u << 2, eNOTIFY = (1u << 3) | eCALLBACK, eDEFAULT = 0 };
}
namespace FilterObj {  // PxFilterObjectType(아래 4 비트) + PxFilterObjectFlag + PxFilterObjectFlagEx(ScActorSim.h:52)
enum : uint32_t {
  eTYPE_RIGID_STATIC = 0, eTYPE_RIGID_DYNAMIC = 1, eTYPE_ARTICULATION = 2, eTYPE_MASK = 15,
  eKINEMATIC = 1u << 4, eTRIGGER = 1u << 5, eCUSTOM_GEOMETRY = 1u << 6,
  eEX_RIGID_STATIC = 1u << 7, eEX_RIGID_DYNAMIC = 1u << 8, eEX_NON_RIGID = 1u << 9
};
}
enum ActorType : int32_t { eRIGID_STATIC = 0, eRIGID_DYNAMIC = 1, eARTICULATION_LINK = 2 };  // PxActorType
enum PairFilteringMode : int32_t { eKEEP = 0, eSUPPRESS = 1, eKILL = 2 };                      // PxPairFilteringMode
// Sc::InteractionType. 조인트(eCONSTRAINTSHADER)·관절체 관절(eARTICULATION) 은 joints·articulation 이 만들지만 행위자 상호작용 목록을 같이 쓴다
// (지우면 마지막을 그 자리로 옮기므로 섞인 순서가 요소 상호작용 순서를 정한다) -> 이 층이 자리표로 들고 있는다.
enum InteractionType : uint8_t { eOVERLAP = 0, eTRIGGER = 1, eMARKER = 2, eCONSTRAINTSHADER = 3, eARTICULATION = 4, eINVALID = 0xff };
namespace IFlag {  // ScInteractionFlags.h:38
enum : uint8_t { eRB_ELEMENT = 1 << 0, eCONSTRAINT = 1 << 1, eFILTERABLE = 1 << 2, eIN_DIRTY_LIST = 1 << 3, eIS_FILTER_PAIR = 1 << 4, eIS_ACTIVE = 1 << 5 };
}
namespace DirtyFlag {  // ScInteractionFlags.h:51
enum : uint8_t { eFILTER_STATE = 1 << 0, eBODY_KINEMATIC = (1 << 1) | eFILTER_STATE, eDOMINANCE = 1 << 2, eREST_OFFSET = 1 << 3, eVISUALIZATION = 1 << 4 };
}
namespace SiFlag {  // ScShapeInteraction.h:76
enum : uint32_t {
  PAIR_FLAGS_MASK = PairFlag::eNEXT_FREE - 1, NEXT_FREE = (PAIR_FLAGS_MASK << 1) & ~PAIR_FLAGS_MASK,
  HAS_TOUCH = NEXT_FREE << 0, HAS_NO_TOUCH = NEXT_FREE << 1, TOUCH_KNOWN = HAS_TOUCH | HAS_NO_TOUCH,
  CONTACTS_COLLECT_POINTS = NEXT_FREE << 2, CONTACTS_RESPONSE_DISABLED = NEXT_FREE << 3,
  CONTACT_FORCE_THRESHOLD_PAIRS = PairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS | PairFlag::eNOTIFY_THRESHOLD_FORCE_LOST,
  CONTACT_REPORT_EVENTS = PairFlag::eNOTIFY_TOUCH_FOUND | PairFlag::eNOTIFY_TOUCH_PERSISTS | PairFlag::eNOTIFY_TOUCH_LOST |
                          PairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS | PairFlag::eNOTIFY_THRESHOLD_FORCE_LOST,
  LL_MANAGER_RECREATE_EVENT = CONTACT_REPORT_EVENTS | CONTACTS_COLLECT_POINTS | CONTACTS_RESPONSE_DISABLED | PairFlag::eMODIFY_CONTACTS
};
}
namespace WuFlag {  // PxcNpWorkUnit.h PxcNpWorkUnitFlag
enum : uint16_t {
  eOUTPUT_CONTACTS = 1 << 0, eOUTPUT_CONSTRAINTS = 1 << 1, eDISABLE_STRONG_FRICTION = 1 << 2, eARTICULATION_BODY0 = 1 << 3, eARTICULATION_BODY1 = 1 << 4,
  eDYNAMIC_BODY0 = 1 << 5, eDYNAMIC_BODY1 = 1 << 6, eSOFT_BODY = 1 << 7, eMODIFIABLE_CONTACT = 1 << 8, eFORCE_THRESHOLD = 1 << 9,
  eDETECT_DISCRETE_CONTACT = 1 << 10, eHAS_KINEMATIC_ACTOR = 1 << 11, eDISABLE_RESPONSE = 1 << 12, eDETECT_CCD_CONTACTS = 1 << 13
};
}
namespace WuStatus {  // PxcNpWorkUnitStatusFlag
enum : uint8_t { eHAS_NO_TOUCH = 1 << 0, eHAS_TOUCH = 1 << 1, eREQUEST_CONSTRAINTS = 1 << 3, eDIRTY_MANAGER = 1 << 5, eREFRESHED_WITH_TOUCH = 1 << 6 };
}
namespace OutStatus {  // PxsContactManagerStatusFlag (PxsContactManagerState.h:50)
enum : uint8_t { eHAS_NO_TOUCH = 1 << 0, eHAS_TOUCH = 1 << 1, eREQUEST_CONSTRAINTS = 1 << 3, eHAS_CCD_RETOUCH = 1 << 4, eDIRTY_MANAGER = 1 << 5,
                 eTOUCH_KNOWN = eHAS_NO_TOUCH | eHAS_TOUCH };
}
static const uint32_t INVALID = 0xffffffffu;
namespace PairRelease {  // ScNPhaseCore.h:75 PairReleaseFlag
enum : uint32_t { eRUN_LOST_TOUCH_LOGIC = 1u << 0, eWAKE_ON_LOST_TOUCH = 1u << 1 };
}
static const uint64_t INVALID_NODE = 0xffffffffull;  // PxNodeIndex() 원값 (mID = PX_INVALID_NODE, mLinkID = 0)
static const uint32_t NEW_CM_MASK = 0x80000000u;  // PxsNphaseCommon.h:40
static const uint32_t CM_BUCKET_BITS = 7;         // PxsNphaseCommon.h:41 (CPU 판 bucket 0)
enum EdgeType : int32_t { eEDGE_CONTACT_MANAGER = 0 };  // IG::Edge::EdgeType (강체·관절체만 -> 늘 eCONTACT_MANAGER, ScInteraction.h getInteractionEdgeType)

// ---- 입력 자료
struct FilterData { uint32_t word0 = 0, word1 = 0, word2 = 0, word3 = 0; };

struct Actor {
  int32_t type = eRIGID_STATIC;      // ActorType
  uint32_t filterAttr = 0;           // Sc::ActorSim::mFilterFlags (종류 + eKINEMATIC + Ex 비트)
  uint32_t actorID = 0;              // Sc::ActorSim::getActorID (swap·잃은 닿음 규칙이 씀)
  uint64_t nodeIndex = INVALID_NODE; // PxNodeIndex::getInd() 원값 (정적은 무효). 섬 갈고리에 그대로 넘긴다
  bool fixedBaseLink = false;        // PxsBodyCore::fixedBaseLink (관절체 뿌리 고정)
  int32_t articulation = -1;         // 관절체 번호 (링크일 때)
  uint32_t linkId = 0, parentLinkId = INVALID;
  bool artDisableSelfCollision = false;
  bool hasConstraints = false;       // ActorSim::BF_HAS_CONSTRAINTS
  uint8_t dominanceGroup = 0;
  float offsetSlop = 0.0f;           // PxsBodyCore::offsetSlop (강체만)
  bool forceStaticKineNotif = false, forceKineKineNotif = false;  // PxRigidBodyFlag::eFORCE_*_NOTIFICATIONS
  // 이 층이 관리
  Vec<int32_t> interactions;  // ActorSim::mInteractions (등록 순서, 지우면 마지막을 그 자리로)
  uint32_t countedInteractions = 0;    // BodySim::registerCountedInteraction 수
  SCHD bool isStatic() const { return (filterAttr & FilterObj::eEX_RIGID_STATIC) != 0; }
  SCHD bool isDynamicRigid() const { return (filterAttr & FilterObj::eEX_RIGID_DYNAMIC) != 0; }
  SCHD bool isKinematic() const { return (filterAttr & FilterObj::eKINEMATIC) != 0; }
};

struct Shape {
  bool valid = false;
  int32_t actor = -1;
  int32_t geomType = 0;              // PxGeometryType
  bool trigger = false;              // PxShapeFlag::eTRIGGER_SHAPE
  FilterData fd;                     // 시뮬레이션 거르기 자료
  float restOffset = 0.0f;
  float torsionalPatchRadius = 0.0f, minTorsionalPatchRadius = 0.0f;
  uint32_t transformCacheId = INVALID;
};

// 조인트로 이은 두 행위자의 충돌 끄기 (joints 모듈이 채움: PxConstraintFlag::eCOLLISION_ENABLED 가 꺼진 조인트)
// ScFiltering.cpp:299 filterJointedBodies -> Scene::findConstraintCore(첫 번째로 찾은 조인트의 플래그)
struct JointPairInfo { bool collisionEnabled; };

// ---- 거르개 (omni.physx 셰이더 등). 반환 = PxFilterFlags, pairFlags 에 쓴다 (PxSimulationFilterShader 와 같은 뜻)
typedef uint32_t (*FilterShaderFn)(uint32_t attr0, const FilterData& fd0, uint32_t attr1, const FilterData& fd1, uint32_t& pairFlags, const void* data);
// PxSimulationFilterCallback::pairFound (없으면 nullptr)
typedef uint32_t (*FilterPairFoundFn)(uint64_t pairID, uint32_t attr0, const FilterData& fd0, int32_t actor0, int32_t shape0, uint32_t attr1,
                                      const FilterData& fd1, int32_t actor1, int32_t shape1, uint32_t& pairFlags, const void* data);
typedef void (*FilterPairLostFn)(uint64_t pairID, uint32_t attr0, const FilterData& fd0, uint32_t attr1, const FilterData& fd1, bool objRemoved,
                                 const void* data);

// ---- 섬 갈고리 (solver 가 구현: PxsSimpleIslandManager.cpp 의 같은 이름 함수)
struct IslandHooks {
  SCHD virtual ~IslandHooks() {}
  SCHDV virtual uint32_t addContactManager(int32_t cm, uint64_t node0, uint64_t node1, int32_t inter, int32_t edgeType) = 0;
  SCHDV virtual void preallocateContactManagers(uint32_t nb, uint32_t* handles) = 0;
  SCHDV virtual bool addPreallocatedContactManager(uint32_t edge, int32_t cm, uint64_t node0, uint64_t node1, int32_t inter, int32_t edgeType) = 0;
  SCHDV virtual void addDelayedDirtyEdges(uint32_t nb, const uint32_t* edges) = 0;  // speculative IslandSim
  SCHDV virtual void setEdgeConnected(uint32_t edge, int32_t edgeType) = 0;
  SCHDV virtual void setEdgeDisconnected(uint32_t edge) = 0;
  SCHDV virtual void removeConnection(uint32_t edge) = 0;
  SCHDV virtual void setEdgeRigidCM(uint32_t edge, int32_t cm) = 0;
  SCHDV virtual void clearEdgeRigidCM(uint32_t edge) = 0;
  SCHDV virtual void deactivateEdge(uint32_t edge) = 0;
  SCHDV virtual bool isSpeculativeNodeActive(uint64_t node) = 0;             // IslandSim::getNode(n).isActive() (추측 섬)
  SCHDV virtual bool isSpeculativeNodeActiveOrActivating(uint64_t node) = 0;
  SCHDV virtual bool isActorActive(int32_t actor) = 0;                        // ActorSim::isActive
  SCHDV virtual void internalWakeUp(int32_t actor) = 0;                        // ActorSim::internalWakeUp
  SCHDV virtual void addToLostTouchList(int32_t actor0, int32_t actor1) = 0;  // Scene::addToLostTouchList
};

// ---- 이 층의 자료
struct Interaction {
  bool alive = false;
  uint8_t type = eINVALID;
  uint8_t iflags = 0;       // IFlag
  uint8_t dirty = 0;        // DirtyFlag (setDirty)
  uint32_t siFlags = 0;     // ShapeInteraction::mFlags (아래 15 비트 = PxPairFlags) / 트리거는 트리거 플래그
  int32_t elem0 = -1, elem1 = -1;   // 저장 순서 (ShapeInteraction 은 shouldSwapBodies 뒤)
  int32_t actor0 = -1, actor1 = -1;
  uint32_t actorSlot0 = INVALID, actorSlot1 = INVALID;  // Interaction::mActorId0/1
  int32_t cm = -1;          // mManager
  uint32_t edge = INVALID;  // mEdgeIndex
  bool sceneRegistered = false;
  uint32_t dirtyPos = INVALID;  // 더러움 목록 안 자리
};

struct ContactManager {  // PxsContactManager + PxcNpWorkUnit 중 좁은 단계·solver 가 읽는 것
  int32_t inter = -1;
  int32_t shape0 = -1, shape1 = -1;
  int32_t geomType0 = 0, geomType1 = 0;
  uint32_t transformCache0 = INVALID, transformCache1 = INVALID;
  float restDistance = 0.0f, torsionalPatchRadius = 0.0f, minTorsionalPatchRadius = 0.0f, offsetSlop = 0.0f;
  uint16_t wuFlags = 0;
  uint8_t dominance0 = 1, dominance1 = 1;
  uint8_t statusFlags = 0;  // PxcNpWorkUnit::mStatusFlags
  uint32_t npIndex = INVALID;
  uint32_t cmFlags = 0;     // PxsContactManager::mFlags (PXS_CM_CHANGEABLE=1, PXS_CM_CCD_LINEAR=2)
  bool inUse = false;
  // 캐시 지움(PxcNpWorkUnit::clearCachedState): 마찰 패치 수 0 (solver 가 읽는다)
  uint32_t cachedStateEpoch = 0;
};

struct NpOutput {  // PxsContactManagerOutput 중 이 층이 쓰는 칸
  uint8_t statusFlag = 0, nbPatches = 0, prevPatches = 0, flags = 0;
};

// 좁은 단계 목록 (PxsContactManagers). 캐시(지속 다양체)는 목록 칸과 함께 움직인다 -> 갈고리로 옮김을 알린다.
struct NpList {
  Vec<int32_t> cms;
  Vec<NpOutput> outputs;
  Vec<int32_t> cacheKind;  // PxsContext::createCache 결과 종류 (0 없음, 1 구 1점, 2 큰 4점, 3 다중)
  SCHD uint32_t size() const { return uint32_t(cms.size()); }
};

struct CacheHooks {  // 좁은 단계 모듈이 들고 있는 칸별 캐시(다양체)를 목록 변경에 맞춰 옮긴다
  SCHD virtual ~CacheHooks() {}
  SCHDV virtual void create(bool newList, uint32_t slot, int32_t geomType0, int32_t geomType1) = 0;  // 새 칸 (목록 끝)
  SCHDV virtual void move(bool newListDst, uint32_t dst, bool newListSrc, uint32_t src) = 0;          // 칸 복사
  SCHDV virtual void destroy(bool newList, uint32_t slot) = 0;
  SCHDV virtual void resize(bool newList, uint32_t n) = 0;
};

// CmPool.h PoolList 의 번호 흉내 (빈 칸 목록 순서 그대로)
struct CmPool {
  uint32_t eltsPerSlab = 256;  // PxSceneDesc::contactPairSlabSize 기본값
  uint32_t slabCount = 0;
  Vec<int32_t> freeList;  // 원본 mFreeList (크기 = mFreeCount)
  Vec<uint8_t> used;
  // CmPool.h:59 preallocate
  SCHD void preallocate(uint32_t nbRequired, int32_t* elements) {
    uint32_t nbToAllocate = nbRequired > freeList.size() ? nbRequired - uint32_t(freeList.size()) : 0;
    uint32_t nbElements = nbRequired - nbToAllocate;
    const uint32_t freeCount = uint32_t(freeList.size());
    for (uint32_t i = 0; i < nbElements; ++i) elements[i] = freeList[freeCount - nbElements + i];
    freeList.resize(freeCount - nbElements);
    if (nbToAllocate) {
      const uint32_t nbSlabs = (nbToAllocate + eltsPerSlab - 1) / eltsPerSlab;
      for (uint32_t s = 0; s < nbSlabs; ++s) {
        slabCount++;
        used.resize(size_t(slabCount) * eltsPerSlab, 0);
        const uint32_t baseIndex = (slabCount - 1) * eltsPerSlab;
        int32_t idx = int32_t(eltsPerSlab - 1);
        for (; idx >= int32_t(nbToAllocate); --idx) freeList.push_back(int32_t(baseIndex + uint32_t(idx)));
        const uint32_t origElements = nbElements;
        int32_t* writeIdx = elements + nbElements;
        for (; idx >= 0; --idx) {
          writeIdx[idx] = int32_t(baseIndex + uint32_t(idx));
          nbElements++;
        }
        nbToAllocate -= (nbElements - origElements);
      }
    }
    for (uint32_t a = 0; a < nbElements; ++a) used[size_t(elements[a])] = 1;
  }
  // CmPool.h:147 get / :183 extend
  SCHD int32_t get() {
    if (freeList.empty()) {
      slabCount++;
      used.resize(size_t(slabCount) * eltsPerSlab, 0);
      const uint32_t baseIndex = (slabCount - 1) * eltsPerSlab;
      for (int32_t i = int32_t(eltsPerSlab - 1); i >= 0; i--) freeList.push_back(int32_t(baseIndex + uint32_t(i)));
    }
    const int32_t e = freeList.back();
    freeList.pop_back();
    used[size_t(e)] = 1;
    return e;
  }
  SCHD void put(int32_t e) {  // CmPool.h:156
    used[size_t(e)] = 0;
    freeList.push_back(e);
  }
  SCHD uint32_t capacity() const { return slabCount * eltsPerSlab; }
};

// 닿음 변화 비트맵 (PxsContext::mContactManagerTouchEvent, 접촉 관리자 번호별)
struct BitSet {
  Vec<uint32_t> w;
  SCHD void growAndSet(uint32_t i) { if ((i >> 5) >= w.size()) w.resize((i >> 5) + 1, 0); w[i >> 5] |= 1u << (i & 31); }
  SCHD void growAndReset(uint32_t i) { if ((i >> 5) >= w.size()) w.resize((i >> 5) + 1, 0); w[i >> 5] &= ~(1u << (i & 31)); }
  SCHD void clear() { for (uint32_t i = 0; i < w.size(); ++i) w[i] = 0u; }
};

// 이번 스텝 사건
struct TouchEvent { int32_t inter; };

class ScPairs {
 public:
  // 장면 (엔진이 채움). shapes 는 ElementSim 번호로 바로 찾는다.
  Vec<Actor> actors;
  Vec<Shape> shapes;
  // 조인트 충돌 표: 두 행위자 번호 (작은 것, 큰 것) -> 충돌 켜짐 여부. 행위자 쌍마다 첫 조인트 (findConstraintCore 가 첫 번째 것을 봄)
  Map<JointPairInfo> jointPairs;
  // 거르기 설정 (Sc::FilteringContext)
  FilterShaderFn filterShader = nullptr;
  const void* filterShaderData = nullptr;
  FilterPairFoundFn filterPairFound = nullptr;
  FilterPairLostFn filterPairLost = nullptr;
  const void* filterCallbackData = nullptr;
  int32_t kineKineFilteringMode = eSUPPRESS, staticKineFilteringMode = eSUPPRESS;  // PxSceneDesc 기본 eDEFAULT = eSUPPRESS
  // 지배 그룹 표 (Scene::getDominanceGroupPair). 기본: 모두 (1,1)
  uint8_t dominance[32][32][2];
  // 좁은 단계 설정
  bool pcm = true;

  IslandHooks* islands = nullptr;
  CacheHooks* caches = nullptr;  // 없으면 캐시 종류만 적는다

  // 이 층의 상태
  Vec<Interaction> inters;
  Vec<int32_t> freeInters;
  Map<int32_t> elementSimMap;  // NPhaseCore::mElementSimMap (찾기 전용)
  Vec<ContactManager> cmsData;                  // 풀 번호로 찾음
  CmPool cmPool;
  NpList npMain, npNew;  // mNarrowPhasePairs, mNewNarrowPhasePairs
  BitSet touchEvent;     // mContactManagerTouchEvent
  uint32_t newTouchCount = 0, lostTouchCount = 0;
  Vec<int32_t> dirtyList;  // NPhaseCore::mDirtyInteractions (PxCoalescedHashSet: 넣은 순서, 지우면 마지막을 그 자리로)
  Vec<TouchEvent> touchFound, touchLost;  // Scene::mTouchFoundEvents / mTouchLostEvents
  // 이번 스텝 사라진 겹침 (processLostContacts 1~3 이 공유: AABBOverlap::mPairUserData 자리)
  struct LostOverlap { int32_t e0, e1, inter; };
  Vec<LostOverlap> lostShape, lostTrigger;

  SCHD ScPairs() {
    for (int i = 0; i < 32; ++i)
      for (int j = 0; j < 32; ++j) dominance[i][j][0] = dominance[i][j][1] = 1;
  }

  // =====================================================================================================
  // 한 스텝의 쌍 처리 (Sc::Scene::simulate 순서). 엔진은 이 순서대로 부른다:
  //   updateDirtyInteractions -> [넓은 단계] -> finishBroadPhase(새 겹침) -> [섬 1차] -> 좁은 단계(목록 두 개) -> mergeNarrowPhase
  //   -> fillTouchEvents -> processNewTouches -> [solver] -> processLostContacts(사라진 겹침) + processNarrowPhaseLostTouchEvents
  //   -> processLostContacts2 -> lostTouchReports/unregisterInteractions -> [섬 3차] -> destroyManagers -> processLostContacts3
  // =====================================================================================================

  // ---- ScNPhaseCore.cpp:886 updateDirtyInteractions (지배 그룹·시각화 전체 더러움은 아직 없음 — BEHAVIOR 에서 안 바뀜)
  SCHD void updateDirtyInteractions() {
    // 목록 스냅샷을 차례로 (PxCoalescedHashSet::getEntries). 도중에 바뀐(convert) 상호작용의 번호는 끝난 뒤에 돌려준다
    // (PhysX 는 다른 풀이라 새 상호작용이 옛 것과 같은 주소일 수 없다 -> "interaction == refInt" 비교가 번호 재사용에 속지 않게).
    deferFree = true;
    const Vec<int32_t> entries = dirtyList;
    for (size_t i = 0; i < entries.size(); ++i) {
      const int32_t it = entries[i];
      int32_t refInt = it;
      Interaction& I0 = inters[size_t(it)];
      if (I0.type <= eMARKER && (I0.dirty & DirtyFlag::eFILTER_STATE)) refInt = refilterInteraction(it);  // needsRefiltering
      if (refInt == it) {
        Interaction& I = inters[size_t(it)];
        if (I.type == eOVERLAP) updateState(it, 0);
        // (eCONSTRAINTSHADER 의 updateState 는 joints 몫)
        I.dirty = 0;  // setClean(false)
        I.iflags &= uint8_t(~IFlag::eIN_DIRTY_LIST);
      }
    }
    dirtyList.clear();
    deferFree = false;
    for (int32_t f : pendingFree) freeInters.push_back(f);
    pendingFree.clear();
  }
  // ActorSim::setActorsInteractionsDirty (ScActorSim.cpp:157): 운동학 전환(ScBodySim.cpp:249 eBODY_KINEMATIC/eFILTERABLE, :275 +eCONSTRAINT),
  // 지배 그룹(ScActorCore.cpp:74), 조인트 끊김(ScConstraintBreakage.cpp:101 eFILTER_STATE/eRB_ELEMENT). other < 0 이면 전부.
  SCHD void setActorsInteractionsDirty(int32_t actor, uint8_t flag, int32_t other, uint8_t interactionFlagMask) {
    const Vec<int32_t> L = actors[size_t(actor)].interactions;
    for (int32_t it : L) {
      const Interaction& I = inters[size_t(it)];
      if ((other < 0 || other == I.actor0 || other == I.actor1) && (I.iflags & interactionFlagMask)) setDirty(it, flag);
    }
  }
  // ShapeSimBase 의 setElementInteractionsDirty (ScShapeSimBase.cpp:65): 거르기 자료 바뀜(eFILTER_STATE/eFILTERABLE), restOffset(eREST_OFFSET/eRB_ELEMENT)
  SCHD void setElementInteractionsDirty(int32_t elem, uint8_t flag, uint8_t interactionFlagMask) {
    const Vec<int32_t> L = actors[size_t(shapes[size_t(elem)].actor)].interactions;
    for (int32_t it : L) {
      const Interaction& I = inters[size_t(it)];
      if (I.type > eMARKER || !(I.elem0 == elem || I.elem1 == elem)) continue;  // ElementInteractionIterator (요소 상호작용만)
      if (I.iflags & interactionFlagMask) setDirty(it, flag);
    }
  }
  // Interaction::setDirty + addToDirtyInteractionList (예: 조인트 끊김 -> ScConstraintBreakage.cpp:96 eFILTER_STATE, solver 가 부름)
  SCHD void setDirty(int32_t it, uint8_t flags) {
    Interaction& I = inters[size_t(it)];
    I.dirty |= flags;
    if (!(I.iflags & IFlag::eIN_DIRTY_LIST)) {
      I.iflags |= IFlag::eIN_DIRTY_LIST;
      I.dirtyPos = uint32_t(dirtyList.size());
      dirtyList.push_back(it);
    }
  }

  // ---- ScPipeline.cpp:477 finishBroadPhase + :684 preallocateContactManagers + :985 postBroadPhaseStage2 의 쌍 부분
  //  triggerPairs/shapePairs = AABB 관리자 겹침 생성 목록(ElementSim 번호, mUserData0/1 순서 그대로)
  SCHD void finishBroadPhase(const int32_t* triggerPairs, uint32_t nbTrigger, const int32_t* shapePairs, uint32_t nbShape) {
    // 트리거: 거르고 바로 만든다 (NPhaseCore::onTriggerOverlapCreated, ScFiltering.cpp:629)
    for (uint32_t i = 0; i < nbTrigger; ++i) {
      const int32_t hi = triggerPairs[2 * i + 1], lo = triggerPairs[2 * i];
      FilterInfo fi;
      bool isTriggerPair;
      filterRbCollisionPair(fi, hi, lo, isTriggerPair, false);
      if (fi.filterFlags & FilterFlag::eKILL) continue;
      createRbElementInteraction(fi, hi, lo, -1, false, isTriggerPair, false);
    }
    preCms.clear();
    preSis.clear();
    preMarkers.clear();
    if (!nbShape) return postBroadPhaseStage2();
    // 거르기 (OverlapFilterTask 64 개씩, 살아남은 쌍을 앞으로 모음 = 전체에서 순서 유지 필터)
    struct Kept { int32_t e0, e1; FilterInfo fi; };
    Vec<Kept> kept;
    Vec<uint32_t> taskKeep, taskSuppress;  // 작업별 수 (묶음 나누기가 이 단위)
    const uint32_t MaxPairs = 64;  // ScPipeline.cpp:444
    for (uint32_t a = 0; a < nbShape; a += MaxPairs) {
      const uint32_t n = nbShape - a < MaxPairs ? nbShape - a : MaxPairs;
      uint32_t keep = 0, supp = 0;
      for (uint32_t i = a; i < a + n; ++i) {
        const int32_t e0 = shapePairs[2 * i], e1 = shapePairs[2 * i + 1];
        FilterInfo fi;
        filterRbCollisionPairAllTests(fi, e0, e1);
        if (!(fi.filterFlags & FilterFlag::eKILL)) {
          if (!(fi.filterFlags & FilterFlag::eSUPPRESS)) keep++;
          else supp++;
          kept.push_back(Kept{e0, e1, fi});
        }
      }
      taskKeep.push_back(keep);
      taskSuppress.push_back(supp);
    }
    uint32_t totalCreated = 0, totalSuppress = 0;
    for (size_t t = 0; t < taskKeep.size(); ++t) { totalCreated += taskKeep[t]; totalSuppress += taskSuppress[t]; }
    preCms.assign(totalCreated, -1);
    preSis.assign(totalCreated, -1);
    preUsedCm.assign(totalCreated, 0);
    preUsedSi.assign(totalCreated, 0);
    preMarkers.assign(totalSuppress, 0);
    // 묶음 (ScPipeline.cpp:772 nbPairsPerTask 256): 거르기 작업 단위로 모아 256 이 넘으면 끊는다. 묶음마다 풀에서 미리 뽑는다.
    const uint32_t nbPairsPerTask = 256;
    uint32_t batchSize = 0, createdStartIdx = 0, suppressedStartIdx = 0, createdCurrIdx = 0, suppressedCurrIdx = 0, createdOverlapCount = 0;
    uint32_t nextCreatedOverlapCount = 0, nextCreatedStartIdx = 0, nextSuppressedStartIdx = 0;
    Vec<Batch> batches;
    for (size_t t = 0; t < taskKeep.size(); ++t) {
      if (!(taskKeep[t] || taskSuppress[t])) continue;
      const uint32_t nb = taskKeep[t] + taskSuppress[t];
      createdOverlapCount += nb;
      batchSize += nb;
      suppressedCurrIdx += taskSuppress[t];
      createdCurrIdx += taskKeep[t];
      if (batchSize >= nbPairsPerTask) {
        batches.push_back(processBatch(nextCreatedOverlapCount, nextCreatedStartIdx, nextSuppressedStartIdx, createdCurrIdx, createdStartIdx,
                                       suppressedCurrIdx, suppressedStartIdx, batchSize));
        nextCreatedOverlapCount = createdOverlapCount;
        nextCreatedStartIdx = createdStartIdx;
        nextSuppressedStartIdx = suppressedStartIdx;
        batchSize = 0;
      }
    }
    if (batchSize)
      batches.push_back(processBatch(nextCreatedOverlapCount, nextCreatedStartIdx, nextSuppressedStartIdx, createdCurrIdx, createdStartIdx,
                                     suppressedCurrIdx, suppressedStartIdx, batchSize));
    // OnOverlapCreatedTask (ScPipeline.cpp:216): 묶음마다 쌍을 차례로 만든다. 모양 순서를 바꿔 넘긴다(pair.mUserData1, mUserData0)
    for (Batch& b : batches) {
      uint32_t curCm = b.cmStart, curSi = b.cmStart, curEi = b.markerStart;
      for (uint32_t i = 0; i < b.nb; ++i) {
        const Kept& k = kept[b.overlapStart + i];
        const int32_t it = createRbElementInteraction(k.fi, k.e1, k.e0, curCm < preCms.size() ? preCms[curCm] : -1, true, false, true,
                                                      curSi < preSis.size() ? preSis[curSi] : -1);
        if (it >= 0) {
          const Interaction& I = inters[size_t(it)];
          if (I.type == eOVERLAP) {
            preSis[curSi] = it;
            preUsedSi[curSi] = 1;
            curSi++;
            if (I.cm >= 0) {
              preUsedCm[curCm] = 1;
              curCm++;
            }
          } else if (I.type == eMARKER) {
            preMarkers[curEi] = it + 1;
            curEi++;
          }
        }
      }
      b.nbShapeInteractions = curSi - b.cmStart;
    }
    lastBatches = batches;
    postBroadPhaseStage2();
  }

  // ---- 좁은 단계 결과 반영 (PxsNphaseImplementationContext.cpp:321 processCms 의 관리 부분). 좁은 단계 모듈이 칸마다 부른다.
  //  statusFlag·nbPatches = 이번 좁은 단계 출력 (PxcDiscreteNarrowPhasePCM 이 쓴 값)
  SCHD void narrowPhaseResult(bool newList, uint32_t slot, uint8_t newStatusFlag, uint8_t nbPatches) {
    NpList& L = newList ? npNew : npMain;
    NpOutput& out = L.outputs[slot];
    ContactManager& cm = cmsData[size_t(L.cms[slot])];
    out.prevPatches = out.nbPatches;
    const uint8_t oldStatusFlag = out.statusFlag;
    const uint8_t oldTouch = oldStatusFlag & OutStatus::eHAS_TOUCH;
    out.statusFlag = newStatusFlag;
    out.nbPatches = nbPatches;
    const uint8_t newTouch = newStatusFlag & OutStatus::eHAS_TOUCH;
    if (newTouch ^ oldTouch) {
      cm.statusFlags = uint8_t(newStatusFlag | (cm.statusFlags & WuStatus::eREFRESHED_WITH_TOUCH));
      touchEvent.growAndSet(uint32_t(L.cms[slot]));
      if (newTouch) newTouchCount++;
      else lostTouchCount++;
    } else if (!(oldStatusFlag & OutStatus::eTOUCH_KNOWN)) {
      cm.statusFlags = uint8_t(newStatusFlag | (cm.statusFlags & WuStatus::eREFRESHED_WITH_TOUCH));
    }
  }
  // PxsNphaseImplementationContext.cpp:599 updateContactManager 의 앞부분: clearManagerTouchEvents
  SCHD void beginNarrowPhase() {
    touchEvent.clear();
    newTouchCount = lostTouchCount = 0;
  }
  // ---- PxsContext.cpp:431 mergeCMDiscreteUpdateResults -> appendContactManagers (:806)
  SCHD void mergeNarrowPhase() {
    const uint32_t existingSize = npMain.size();
    const uint32_t nbToAdd = npNew.size();
    for (uint32_t a = 0; a < nbToAdd; ++a) {
      npMain.cms.push_back(npNew.cms[a]);
      npMain.outputs.push_back(npNew.outputs[a]);
      npMain.cacheKind.push_back(npNew.cacheKind[a]);
    }
    if (caches) {
      caches->resize(false, existingSize + nbToAdd);
      for (uint32_t a = 0; a < nbToAdd; ++a) caches->move(false, existingSize + a, true, a);
    }
    for (uint32_t a = 0; a < nbToAdd; ++a) {
      ContactManager& cm = cmsData[size_t(npNew.cms[a])];
      cm.npIndex = (existingSize + a) << CM_BUCKET_BITS;
      if (cm.statusFlags & WuStatus::eREFRESHED_WITH_TOUCH) cm.statusFlags &= uint8_t(~WuStatus::eREFRESHED_WITH_TOUCH);  // processPartitionEdges 는 GPU 전용
    }
    npNew.cms.clear();
    npNew.outputs.clear();
    npNew.cacheKind.clear();
    if (caches) caches->resize(true, 0);
  }
  // ---- PxsContext.cpp:528 fillManagerTouchEvents (접촉 관리자 번호 순서)
  SCHD void fillTouchEvents() {
    touchFound.clear();
    touchLost.clear();
    for (uint32_t w = 0; w < touchEvent.w.size(); ++w)
      for (uint32_t b = touchEvent.w[w]; b; b &= b - 1) {
        const uint32_t index = (w << 5) | ctz32(b);
        const ContactManager& cm = cmsData[index];
        if (cm.statusFlags & WuStatus::eHAS_TOUCH) touchFound.push_back(TouchEvent{cm.inter});  // getTouchStatus (CCD 재닿음 없음)
        else touchLost.push_back(TouchEvent{cm.inter});
      }
  }
  // ---- ScPipeline.cpp:98 InteractionNewTouchTask: managerNewTouch (보고 쌍 처리는 없음) + :1726 setEdgesConnected
  SCHD void processNewTouches() {
    for (const TouchEvent& e : touchFound) {
      Interaction& I = inters[size_t(e.inter)];
      if (!(I.siFlags & SiFlag::HAS_TOUCH)) {  // ScShapeInteraction.cpp:661
        I.siFlags &= ~SiFlag::HAS_NO_TOUCH;
        I.siFlags |= SiFlag::HAS_TOUCH;
      }
    }
  }
  SCHD void setEdgesConnected() {
    for (const TouchEvent& e : touchFound) {
      const Interaction& I = inters[size_t(e.inter)];
      if (I.edge == INVALID) continue;
      if (!(I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED)) islands->setEdgeConnected(I.edge, eEDGE_CONTACT_MANAGER);
    }
  }
  // ---- ScPipeline.cpp:2147 processLostContacts: 사라진 겹침의 상호작용 찾기. 잃은 닿음 섬 처리(:2220)도 여기서
  SCHD void processLostContacts(const int32_t* shapePairs, uint32_t nbShape, const int32_t* triggerPairs, uint32_t nbTrigger) {
    lostShape.clear();
    lostTrigger.clear();
    for (uint32_t i = 0; i < nbShape; ++i) lostShape.push_back(LostOverlap{shapePairs[2 * i], shapePairs[2 * i + 1], findInteraction(shapePairs[2 * i], shapePairs[2 * i + 1])});
    for (uint32_t i = 0; i < nbTrigger; ++i) lostTrigger.push_back(LostOverlap{triggerPairs[2 * i], triggerPairs[2 * i + 1], -1});
  }
  SCHD void processNarrowPhaseLostTouchEventsIslands() {  // ScPipeline.cpp:2220
    for (const TouchEvent& e : touchLost) {
      const Interaction& I = inters[size_t(e.inter)];
      if (I.edge == INVALID) continue;
      islands->setEdgeDisconnected(I.edge);
    }
  }
  SCHD void processNarrowPhaseLostTouchEvents() {  // ScPipeline.cpp:2253
    for (const TouchEvent& e : touchLost) {
      Interaction& I = inters[size_t(e.inter)];
      if (I.edge == INVALID) continue;
      if (managerLostTouch(e.inter) && !(I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED)) islands->addToLostTouchList(I.actor0, I.actor1);
    }
  }
  // ScPipeline.cpp:2283 processLostContacts2: 섬 간선 떼기
  SCHD void processLostContacts2() {
    for (const LostOverlap& p : lostShape)
      if (p.inter >= 0 && inters[size_t(p.inter)].type == eOVERLAP) clearIslandGenData(p.inter);
  }
  // ScPipeline.cpp:2324 lostTouchReports (보고 없음: 깨우기 규칙만)
  SCHD void lostTouchReports() {
    for (const LostOverlap& p : lostShape)
      if (p.inter >= 0 && inters[size_t(p.inter)].type == eOVERLAP) lostTouchReportsOne(p.inter, /*wakeOnLostTouch*/ true, -1);
  }
  // ScPipeline.cpp:2352 unregisterInteractions (Scene 목록, 순서 영향 없음 — 표시만)
  SCHD void unregisterInteractions() {
    for (const LostOverlap& p : lostShape)
      if (p.inter >= 0) {
        Interaction& I = inters[size_t(p.inter)];
        if (I.type == eOVERLAP || I.type == eMARKER) I.sceneRegistered = false;
      }
  }
  // ScPipeline.cpp:2374 destroyManagers (섬 3차 뒤)
  SCHD void destroyManagers() {
    for (const LostOverlap& p : lostShape)
      if (p.inter >= 0) {
        Interaction& I = inters[size_t(p.inter)];
        if (I.type == eOVERLAP && I.cm >= 0) destroyManager(p.inter);
      }
  }
  // ScPipeline.cpp:2403 processLostContacts3: onOverlapRemoved -> releaseElementPair
  SCHD void processLostContacts3() {
    for (const LostOverlap& p : lostShape) onOverlapRemoved(p.e0, p.e1, p.inter);
    for (const LostOverlap& p : lostTrigger) onOverlapRemoved(p.e0, p.e1, -1);
  }

  // ---- 행위자 활성화·비활성화 (Sc::Scene::wakeObjectsUp / putInteractionsToSleep 가 상호작용마다 부르는 것). solver 가 부른다.
  SCHD bool activateInteraction(int32_t it) {  // ScInteraction activateInteraction -> ShapeInteraction::onActivate(NULL)
    Interaction& I = inters[size_t(it)];
    if (I.type != eOVERLAP) return false;
    return onActivate(it, -1);
  }
  SCHD bool deactivateInteraction(int32_t it) { return onDeactivate(it); }

  // ---- 조인트 끊김 등으로 캐시 지움 (ShapeInteraction::resetManagerCachedState, ScShapeInteraction.cpp:194). updateState 가 부른다.

  // ---- 모양 빼기 (API: 행위자·모양 제거) = NPhaseCore::onVolumeRemoved (ScNPhaseCore.cpp:109), ShapeSimBase::removeFromBroadPhase (ScShapeSimBase.cpp:173).
  // 행위자의 상호작용 목록을 뒤에서부터 보며 이 모양이 든 요소 상호작용을 푼다 (ElementInteractionReverseIterator, ScElementSim.cpp:62).
  // 행위자 제거는 모양 순서대로 (Scene::removeShapes, ScScene.cpp:2285). 넓은 단계에서 빼는 것은 AABB 관리자 몫.
  SCHD void onVolumeRemoved(int32_t elem, bool wakeOnLostTouch) {
    const uint32_t flags = PairRelease::eRUN_LOST_TOUCH_LOGIC | (wakeOnLostTouch ? PairRelease::eWAKE_ON_LOST_TOUCH : 0u);
    const int32_t actor = shapes[size_t(elem)].actor;
    size_t last = actors[size_t(actor)].interactions.size();
    while (last > 0) {
      --last;
      const Vec<int32_t>& L = actors[size_t(actor)].interactions;
      if (last >= L.size()) continue;
      const int32_t it = L[last];
      const Interaction& I = inters[size_t(it)];
      if (I.type > eMARKER) continue;  // 요소 상호작용만 (eRB_ELEMENT)
      if (I.elem0 == elem || I.elem1 == elem) releaseElementPair(it, flags, elem);
    }
    shapes[size_t(elem)].valid = false;
  }

  // ---- joints·articulation 의 상호작용 자리표 (Interaction::registerInActors: 행위자 0 다음 1). 돌려준 번호로 지운다.
  SCHD int32_t addExternalInteraction(int32_t actor0, int32_t actor1, uint8_t type) {
    const int32_t it = allocInteraction();
    Interaction& I = inters[size_t(it)];
    I.alive = true;
    I.type = type;
    I.iflags = type == eCONSTRAINTSHADER ? IFlag::eCONSTRAINT : 0;  // ConstraintInteraction: eCONSTRAINT, ArticulationJointSim: 없음
    I.actor0 = actor0;
    I.actor1 = actor1;
    if (actor0 >= 0) registerInActor(actor0, it, 0);
    if (actor1 >= 0) registerInActor(actor1, it, 1);
    return it;
  }
  SCHD void removeExternalInteraction(int32_t it) {  // ConstraintInteraction::destroy (ScConstraintInteraction.cpp:66): 더러움 목록에서 빼고(setClean(true)) 행위자 목록에서 뺌
    Interaction& I = inters[size_t(it)];
    if (I.iflags & IFlag::eIN_DIRTY_LIST) {
      const uint32_t pos = inters[size_t(it)].dirtyPos;
      const int32_t last = dirtyList.back();
      dirtyList[pos] = last;
      inters[size_t(last)].dirtyPos = pos;
      dirtyList.pop_back();
      inters[size_t(it)].dirtyPos = INVALID;
      I.iflags &= uint8_t(~IFlag::eIN_DIRTY_LIST);
    }
    I.dirty = 0;
    if (I.actor0 >= 0) unregisterFromActor(I.actor0, it, 0);
    if (I.actor1 >= 0) unregisterFromActor(I.actor1, it, 1);
    freeInteraction(it);
  }
  // 시험·장면 적재용: 이미 있는 목록 그대로 채우기 (행위자 목록 끝에 붙이고 칸 번호를 적는다)
  SCHD void appendToActorList(int32_t actor, int32_t it) {
    const Interaction& I = inters[size_t(it)];
    registerInActor(actor, it, I.actor0 == actor ? 0 : 1);
  }
  SCHD int32_t newInteractionRecord(int32_t actor0, int32_t actor1, uint8_t type) {
    const int32_t it = allocInteraction();
    Interaction& I = inters[size_t(it)];
    I.alive = true;
    I.type = type;
    I.iflags = type == eCONSTRAINTSHADER ? IFlag::eCONSTRAINT : 0;
    I.actor0 = actor0;
    I.actor1 = actor1;
    return it;
  }

  // ---- 사용자 API 가 부른 관리자 다시 등록 (자세 set -> ShapeSimBase::onVolumeOrTransformChange -> ShapeInteraction::resetManagerCachedState).
  // 장면 적재(core/scene/pairs_log.h)·그림자 시험이 쓴다 (리드 09-30 추가, 안쪽 함수는 그대로).
  SCHD void userResetManagerCachedState(int32_t it) { resetManagerCachedState(it); }

  // ---- 조회
  SCHD int32_t findInteraction(int32_t e0, int32_t e1) const {
    const int32_t* it = elementSimMap.findPtr(key(e0, e1));
    return it ? *it : -1;
  }

 private:
  struct FilterInfo {
    uint32_t filterFlags = 0, pairFlags = 0;
    bool hasPairID = false;
  };
  struct Batch {
    uint32_t overlapStart, cmStart, markerStart, nb, nbShapeInteractions;
  };
  Vec<int32_t> preCms, preSis;  // mPreallocatedContactManagers / mPreallocatedShapeInteractions
  Vec<uint8_t> preUsedCm, preUsedSi;
  Vec<int32_t> preMarkers;      // 쓰인 표시 = 0 아님
  Vec<Batch> lastBatches;

  SCHD static uint64_t key(int32_t a, int32_t b) {  // ElementSimKey (작은 번호가 앞)
    uint32_t x = uint32_t(a), y = uint32_t(b);
    if (x > y) { const uint32_t t = x; x = y; y = t; }
    return (uint64_t(x) << 32) | y;
  }
  SCHD static uint64_t pairID(int32_t a, int32_t b) {  // ScFiltering.cpp:45 getPairID
    uint64_t x = uint32_t(a), y = uint32_t(b);
    if (y < x) { const uint64_t t = x; x = y; y = t; }
    return (x << 32) | y;
  }
  SCHD uint32_t filterAttrOf(int32_t e, bool supportTriggers) const {  // ScFiltering.cpp:58
    const Shape& s = shapes[size_t(e)];
    uint32_t a = actors[size_t(s.actor)].filterAttr;
    if (supportTriggers && s.trigger) a |= FilterObj::eTRIGGER;
    return a;
  }

  // ScPipeline.cpp:332 processBatch: 풀에서 관리자·상호작용·표식을 미리 뽑는다
  SCHD Batch processBatch(uint32_t nextCreatedOverlapCount, uint32_t nextCreatedStartIdx, uint32_t nextSuppressedStartIdx, uint32_t createdCurrIdx,
                     uint32_t& createdStartIdx, uint32_t suppressedCurrIdx, uint32_t& suppressedStartIdx, uint32_t batchSize) {
    Batch b{nextCreatedOverlapCount, nextCreatedStartIdx, nextSuppressedStartIdx, batchSize, 0};
    const uint32_t nbToCreate = createdCurrIdx - createdStartIdx;
    const uint32_t nbToSuppress = suppressedCurrIdx - suppressedStartIdx;
    if (nbToCreate) cmPool.preallocate(nbToCreate, preCms.data() + createdStartIdx);
    if (cmsData.size() < cmPool.capacity()) cmsData.resize(cmPool.capacity());
    for (uint32_t i = 0; i < nbToCreate; ++i) preSis[createdStartIdx + i] = allocInteraction();
    (void)nbToSuppress;
    createdStartIdx = createdCurrIdx;
    suppressedStartIdx = suppressedCurrIdx;
    return b;
  }

  // ScPipeline.cpp:985 postBroadPhaseStage2 의 쌍 부분 (잃은 닿음 쌍 처리는 solver)
  SCHD void postBroadPhaseStage2() {
    // registerContactManagers (:1176): 미리 뽑은 관리자 중 쓰인 것을 순서대로 좁은 단계 새 목록에
    for (size_t a = 0; a < preCms.size(); ++a)
      if (preUsedCm[a]) registerContactManager(preCms[a], 0, 0);
    // registerInteractions (:1203): 행위자 목록 (쓰인 상호작용, 순서대로) + 셈
    for (size_t a = 0; a < preSis.size(); ++a)
      if (preUsedSi[a]) {
        const int32_t it = preSis[a];
        Interaction& I = inters[size_t(it)];
        registerInActor(I.actor0, it, 0);
        registerInActor(I.actor1, it, 1);
        if (actors[size_t(I.actor0)].isDynamicRigid()) actors[size_t(I.actor0)].countedInteractions++;
        if (actors[size_t(I.actor1)].isDynamicRigid()) actors[size_t(I.actor1)].countedInteractions++;
      }
    for (size_t a = 0; a < preMarkers.size(); ++a)
      if (preMarkers[a]) {
        const int32_t it = preMarkers[a] - 1;
        registerInActor(inters[size_t(it)].actor0, it, 0);
        registerInActor(inters[size_t(it)].actor1, it, 1);
      }
    // registerSceneInteractions (:1227)
    for (size_t a = 0; a < preSis.size(); ++a)
      if (preUsedSi[a]) inters[size_t(preSis[a])].sceneRegistered = true;
    for (size_t a = 0; a < preMarkers.size(); ++a)
      if (preMarkers[a]) inters[size_t(preMarkers[a] - 1)].sceneRegistered = true;
    // islandInsertion 준비 (:1020-1070): 상호작용 수만큼 간선 번호를 미리 받고, 묶음 순서로 addPreallocatedContactManager
    uint32_t total = 0;
    for (const Batch& b : lastBatches) total += b.nbShapeInteractions;
    if (total) {
      Vec<uint32_t> handles(total);
      islands->preallocateContactManagers(total, handles.data());
      uint32_t h = 0;
      Vec<uint32_t> delayed;
      for (const Batch& b : lastBatches) {
        if (!b.nbShapeInteractions) continue;
        delayed.clear();
        for (uint32_t a = 0; a < b.nbShapeInteractions; ++a) {
          if (!preUsedSi[b.cmStart + a]) continue;
          const int32_t it = preSis[b.cmStart + a];
          Interaction& I = inters[size_t(it)];
          const Actor& a0 = actors[size_t(I.actor0)];
          const Actor& a1 = actors[size_t(I.actor1)];
          const uint64_t nodeB = a1.isStatic() ? INVALID_NODE : a1.nodeIndex;
          const uint32_t edgeIdx = handles[h++];
          const bool isDirty = islands->addPreallocatedContactManager(edgeIdx, I.cm, a0.nodeIndex, nodeB, it, eEDGE_CONTACT_MANAGER);
          if (isDirty) delayed.push_back(edgeIdx);
          I.edge = edgeIdx;
        }
        if (!delayed.empty()) delayedDirty.push_back(delayed);
        else delayedDirty.push_back({});
      }
      // islandInsertion (:1110): 작업별 지연 간선을 차례로
      for (auto& d : delayedDirty)
        if (!d.empty()) islands->addDelayedDirtyEdges(uint32_t(d.size()), d.data());
      delayedDirty.clear();
    }
    // 쓰이지 않은 미리 뽑은 관리자를 풀에 되돌림 (:1076, 배열 순서)
    for (size_t a = 0; a < preCms.size(); ++a)
      if (!preUsedCm[a]) cmPool.put(preCms[a]);
    for (size_t a = 0; a < preSis.size(); ++a)
      if (!preUsedSi[a] && preSis[a] >= 0) freeInteraction(preSis[a]);
    preCms.clear();
    preSis.clear();
    preUsedCm.clear();
    preUsedSi.clear();
    preMarkers.clear();
    lastBatches.clear();
  }
  Vec<Vec<uint32_t>> delayedDirty;

  SCHD int32_t allocInteraction() {
    int32_t it;
    if (!freeInters.empty()) {
      it = freeInters.back();
      freeInters.pop_back();
    } else {
      it = int32_t(inters.size());
      inters.push_back(Interaction());
    }
    inters[size_t(it)] = Interaction();
    return it;
  }
  SCHD void freeInteraction(int32_t it) {
    inters[size_t(it)].alive = false;
    if (deferFree) pendingFree.push_back(it);
    else freeInters.push_back(it);
  }
  SCHD void registerInActor(int32_t actor, int32_t it, int which) {  // ScActorSim.cpp:101
    Actor& A = actors[size_t(actor)];
    const uint32_t id = uint32_t(A.interactions.size());
    A.interactions.push_back(it);
    (which == 0 ? inters[size_t(it)].actorSlot0 : inters[size_t(it)].actorSlot1) = id;
  }
  SCHD void unregisterFromActor(int32_t actor, int32_t it, int which) {  // ScActorSim.cpp:108
    Actor& A = actors[size_t(actor)];
    const uint32_t i = which == 0 ? inters[size_t(it)].actorSlot0 : inters[size_t(it)].actorSlot1;
    A.interactions[i] = A.interactions.back();
    A.interactions.pop_back();
    if (i < A.interactions.size()) {
      Interaction& J = inters[size_t(A.interactions[i])];
      // Interaction::setActorId: 이 행위자가 J 의 몇 번째 행위자인지
      if (J.actor0 == actor) J.actorSlot0 = i;
      else J.actorSlot1 = i;
    }
  }
  SCHD void unregisterFromActors(int32_t it) {  // Interaction::unregisterFromActors: 행위자 0 먼저
    const Interaction& I = inters[size_t(it)];
    unregisterFromActor(I.actor0, it, 0);
    unregisterFromActor(I.actor1, it, 1);
  }

  // ---- 거르기 (ScFiltering.cpp)
  SCHD static bool filterObjectIsKinematic(uint32_t a) { return (a & FilterObj::eKINEMATIC) != 0; }
  SCHD static bool filterObjectIsTrigger(uint32_t a) { return (a & FilterObj::eTRIGGER) != 0; }
  SCHD static uint32_t filterType(uint32_t a) { return a & FilterObj::eTYPE_MASK; }
  SCHD static void checkFilterFlags(uint32_t& f) {  // :115
    if ((f & (FilterFlag::eKILL | FilterFlag::eSUPPRESS)) == (FilterFlag::eKILL | FilterFlag::eSUPPRESS)) f &= ~FilterFlag::eKILL;
  }
  SCHD static uint32_t checkRbPairFlags(bool isKinePair, uint32_t pairFlags, uint32_t filterFlags, bool isNonRigid) {  // :137
    if (filterFlags & (FilterFlag::eSUPPRESS | FilterFlag::eKILL)) return pairFlags;
    if (isKinePair && (pairFlags & PairFlag::eSOLVE_CONTACT)) pairFlags &= ~PairFlag::eSOLVE_CONTACT;
    if (isNonRigid && (pairFlags & PairFlag::eDETECT_CCD_CONTACT)) pairFlags &= ~PairFlag::eDETECT_CCD_CONTACT;
    return pairFlags;
  }
  SCHD void filterSecondStage(FilterInfo& fi, int32_t s0, int32_t s1, bool isKinePair, uint32_t fa0, uint32_t fa1, bool runCallbacks, bool isNonRigid) {  // :184
    const FilterData& fd0 = shapes[size_t(s0)].fd;
    const FilterData& fd1 = shapes[size_t(s1)].fd;
    fi.filterFlags = filterShader(fa0, fd0, fa1, fd1, fi.pairFlags, filterShaderData);
    if (fi.filterFlags & FilterFlag::eCALLBACK) {
      if (filterPairFound) {
        if (!runCallbacks) return;
        fi.filterFlags = filterPairFound(pairID(s0, s1), fa0, fd0, shapes[size_t(s0)].actor, s0, fa1, fd1, shapes[size_t(s1)].actor, s1, fi.pairFlags,
                                         filterCallbackData);
        fi.hasPairID = true;
      } else {
        fi.filterFlags &= ~FilterFlag::eNOTIFY;
      }
    }
    checkFilterFlags(fi.filterFlags);
    const bool hasNotify = (fi.filterFlags & FilterFlag::eNOTIFY) == FilterFlag::eNOTIFY;
    const bool hasKill = (fi.filterFlags & FilterFlag::eKILL) != 0;
    if (fi.hasPairID && (hasKill || !hasNotify)) {
      if (hasKill && hasNotify && filterPairLost) filterPairLost(pairID(s0, s1), fa0, fd0, fa1, fd1, false, filterCallbackData);
      if (!hasNotify) fi.filterFlags &= ~FilterFlag::eNOTIFY;
      fi.hasPairID = false;
    }
    if (runCallbacks || !(fi.filterFlags & FilterFlag::eCALLBACK)) fi.pairFlags = checkRbPairFlags(isKinePair, fi.pairFlags, fi.filterFlags, isNonRigid);
  }
  SCHD bool filterArticulationLinks(const Actor& b0, const Actor& b1) const {  // :276
    if (b0.articulation == b1.articulation) {
      if (b0.artDisableSelfCollision) return true;
      if (b1.linkId < b0.linkId) return b0.parentLinkId == b1.linkId;
      return b1.parentLinkId == b0.linkId;
    }
    return false;
  }
  SCHD bool filterJointedBodies(int32_t a0, int32_t a1) const {  // :299
    const Actor& A0 = actors[size_t(a0)];
    const Actor& A1 = actors[size_t(a1)];
    if (!A0.hasConstraints && !A1.hasConstraints) return false;
    const JointPairInfo* it = jointPairs.findPtr(key(a0, a1));
    return it ? !it->collisionEnabled : false;
  }
  SCHD static bool validateSuppress(const Actor* b0, const Actor* b1, bool staticKine) {  // :324
    if (b0 && (staticKine ? b0->forceStaticKineNotif : b0->forceKineKineNotif)) return false;
    if (b1 && (staticKine ? b1->forceStaticKineNotif : b1->forceKineKineNotif)) return false;
    return true;
  }
  SCHD bool filterKinematics(const Actor* b0, const Actor* b1, bool kine0, bool kine1) const {  // :335
    if (kine0 | kine1) {
      if (staticKineFilteringMode != eKEEP)
        if (!b0 || !b1) return validateSuppress(b0, b1, true);
      if (kineKineFilteringMode != eKEEP)
        if (kine0 && kine1) return validateSuppress(b0, b1, false);
    }
    return false;
  }
  template <bool runAllTests>
  SCHD bool filterShared(FilterInfo& fi, bool& isNonRigid, bool& isKinePair, int32_t s0, int32_t s1, uint32_t fa0, uint32_t fa1) {  // :356
    const bool kine0 = filterObjectIsKinematic(fa0), kine1 = filterObjectIsKinematic(fa1);
    const int32_t a0 = shapes[size_t(s0)].actor, a1 = shapes[size_t(s1)].actor;
    const Actor* bs0 = (fa0 & FilterObj::eEX_RIGID_DYNAMIC) ? &actors[size_t(a0)] : nullptr;
    const Actor* bs1 = (fa1 & FilterObj::eEX_RIGID_DYNAMIC) ? &actors[size_t(a1)] : nullptr;
    if (!isNonRigid && filterKinematics(bs0, bs1, kine0, kine1)) { fi = FilterInfo{FilterFlag::eSUPPRESS, 0, false}; return true; }
    if (filterJointedBodies(a0, a1)) { fi = FilterInfo{FilterFlag::eSUPPRESS, 0, false}; return true; }
    if (!isNonRigid && shapes[size_t(s0)].geomType == 8 && shapes[size_t(s1)].geomType == 8) {  // 삼각메시끼리 (SDF 없음)
      fi = FilterInfo{FilterFlag::eKILL, 0, false};
      return true;
    }
    const uint32_t t0 = filterType(fa0), t1 = filterType(fa1);
    const bool link0 = t0 == FilterObj::eTYPE_ARTICULATION, link1 = t1 == FilterObj::eTYPE_ARTICULATION;
    if (runAllTests) {
      if (link0 ^ link1) {
        if (link0) {
          const bool isStaticOrKinematic = (t1 == FilterObj::eTYPE_RIGID_STATIC) || kine1;
          if (bs0->fixedBaseLink && isStaticOrKinematic) { fi = FilterInfo{FilterFlag::eSUPPRESS, 0, false}; return true; }
        }
        if (link1) {
          const bool isStaticOrKinematic = (t0 == FilterObj::eTYPE_RIGID_STATIC) || kine0;
          if (bs1->fixedBaseLink && isStaticOrKinematic) { fi = FilterInfo{FilterFlag::eSUPPRESS, 0, false}; return true; }
        }
      }
    }
    if (link0 && link1) {
      if (runAllTests)
        if (bs0->fixedBaseLink && bs1->fixedBaseLink) { fi = FilterInfo{FilterFlag::eSUPPRESS, 0, false}; return true; }
      if (filterArticulationLinks(*bs0, *bs1)) { fi = FilterInfo{FilterFlag::eKILL, 0, false}; return true; }
    }
    isKinePair = kine0 && kine1;
    return false;
  }
  SCHD void filterRbCollisionPair(FilterInfo& fi, int32_t s0, int32_t s1, bool& isTriggerPair, bool runCallbacks) {  // :467
    const uint32_t fa0 = filterAttrOf(s0, true), fa1 = filterAttrOf(s1, true);
    const bool trigger0 = filterObjectIsTrigger(fa0), trigger1 = filterObjectIsTrigger(fa1);
    isTriggerPair = trigger0 || trigger1;
    bool isNonRigid = false, isKinePair = false;
    if (isTriggerPair) {
      if (trigger0 && trigger1) { fi = FilterInfo{FilterFlag::eKILL, 0, false}; return; }
      isKinePair = filterObjectIsKinematic(fa0) && filterObjectIsKinematic(fa1);
    } else {
      if (filterShared<false>(fi, isNonRigid, isKinePair, s0, s1, fa0, fa1)) return;
    }
    filterSecondStage(fi, s0, s1, isKinePair, fa0, fa1, runCallbacks, isNonRigid);
  }
  SCHD void filterRbCollisionPairAllTests(FilterInfo& fi, int32_t s0, int32_t s1) {  // :501 (runOverlapFilters :586 가 부름)
    fi = FilterInfo();
    const uint32_t fa0 = filterAttrOf(s0, false), fa1 = filterAttrOf(s1, false);
    bool isNonRigid = false, isKinePair = false;
    if (filterShared<true>(fi, isNonRigid, isKinePair, s0, s1, fa0, fa1)) return;
    filterSecondStage(fi, s0, s1, isKinePair, fa0, fa1, true, isNonRigid);
  }

  // ---- 재거르기 (ScFiltering.cpp:669 refilterInteraction, 사용자 정보 없음 경로) / ScNPhaseCore.cpp:424 convert
  SCHD int32_t refilterInteraction(int32_t it) {
    Interaction& I = inters[size_t(it)];
    const int32_t s0 = I.elem0, s1 = I.elem1;
    if ((I.iflags & IFlag::eIS_FILTER_PAIR) && filterPairLost)
      filterPairLost(pairID(s0, s1), filterAttrOf(s0, true), shapes[size_t(s0)].fd, filterAttrOf(s1, true), shapes[size_t(s1)].fd, false, filterCallbackData);
    FilterInfo fi;
    bool isTriggerPair;
    filterRbCollisionPair(fi, s0, s1, isTriggerPair, true);
    if ((I.iflags & IFlag::eIS_FILTER_PAIR) && (fi.filterFlags & FilterFlag::eNOTIFY) != FilterFlag::eNOTIFY) {
      I.iflags &= uint8_t(~IFlag::eIS_FILTER_PAIR);
      fi.hasPairID = false;
    }
    uint8_t newType;
    if (fi.filterFlags & FilterFlag::eKILL) newType = eINVALID;
    else if (fi.filterFlags & FilterFlag::eSUPPRESS) newType = eMARKER;
    else if (shapes[size_t(s0)].trigger || shapes[size_t(s1)].trigger) newType = eTRIGGER;
    else newType = eOVERLAP;
    if (I.type != newType) return convert(it, newType, fi);
    if (I.type == eOVERLAP) setPairFlags(I, fi.pairFlags);  // 보고 쌍 목록 처리 없음
    else if (I.type == eTRIGGER) I.siFlags = fi.pairFlags;
    return it;
  }
  SCHD int32_t convert(int32_t it, uint8_t newType, FilterInfo& fi) {
    Interaction& I = inters[size_t(it)];
    const int32_t eA = I.elem0, eB = I.elem1;
    const int32_t a0 = I.actor0, a1 = I.actor1;
    if (actors[size_t(a0)].type == eRIGID_DYNAMIC && !islands->isActorActive(a0)) islands->internalWakeUp(a0);
    if (actors[size_t(a1)].type == eRIGID_DYNAMIC && !islands->isActorActive(a1)) islands->internalWakeUp(a1);
    I.iflags &= uint8_t(~IFlag::eIS_FILTER_PAIR);
    releaseElementPair(it, PairRelease::eWAKE_ON_LOST_TOUCH | PairRelease::eRUN_LOST_TOUCH_LOGIC, -1, /*removeFromDirtyList*/ false);
    int32_t result = -1;
    if (newType == eMARKER) result = createMarker(eA, eB, false);
    else if (newType == eOVERLAP) result = createShapeInteraction(eA, eB, fi.pairFlags, -1, -1);
    else if (newType == eTRIGGER) result = createTriggerInteraction(eA, eB, fi.pairFlags);
    if (fi.hasPairID && result >= 0) inters[size_t(result)].iflags |= IFlag::eIS_FILTER_PAIR;
    return result;
  }
  bool deferFree = false;
  Vec<int32_t> pendingFree;

  // ---- 상호작용 만들기 (ScNPhaseCore.cpp:132, :182, :260, :297)
  SCHD bool shouldSwapBodies(int32_t s0, int32_t s1) const {
    const Actor& rs0 = actors[size_t(shapes[size_t(s0)].actor)];
    if (rs0.type == eRIGID_STATIC) return true;
    const Actor& rs1 = actors[size_t(shapes[size_t(s1)].actor)];
    const bool isDyna0 = rs0.type == eRIGID_DYNAMIC, isDyna1 = rs1.type == eRIGID_DYNAMIC;
    if (rs0.type == eARTICULATION_LINK) {
      if (isDyna1 || rs1.type == eARTICULATION_LINK)
        if (rs0.fixedBaseLink) return true;
    } else if (isDyna0) {
      if (rs1.type == eARTICULATION_LINK)
        if (!rs1.fixedBaseLink) return true;
    }
    if (isDyna1 && isDyna0 && rs0.isKinematic()) return true;
    if (rs0.type == rs1.type && rs0.actorID < rs1.actorID) {
      const bool actorBKinematic = isDyna1 && rs1.isKinematic();
      if (!actorBKinematic) return true;
    }
    return false;
  }
  // parallelCreate = OnOverlapCreatedTask 경로 (관리자·상호작용을 미리 뽑아 둠, 등록은 나중에 몰아서)
  SCHD int32_t createRbElementInteraction(const FilterInfo& fi, int32_t s0, int32_t s1, int32_t preCm, bool parallelCreate, bool isTriggerPair,
                                     bool fromOverlapTask, int32_t preSi = -1) {
    int32_t it;
    if (!(fi.filterFlags & FilterFlag::eSUPPRESS)) {
      if (!isTriggerPair) it = createShapeInteraction(s0, s1, fi.pairFlags, preCm, parallelCreate ? preSi : -1);
      else it = createTriggerInteraction(s0, s1, fi.pairFlags);
    } else {
      it = createMarker(s0, s1, fromOverlapTask);
    }
    if (fi.hasPairID) inters[size_t(it)].iflags |= IFlag::eIS_FILTER_PAIR;
    return it;
  }
  SCHD int32_t createShapeInteraction(int32_t s0, int32_t s1, uint32_t pairFlags, int32_t preCm, int32_t preSi) {  // :260
    if (shouldSwapBodies(s0, s1)) { const int32_t t = s0; s0 = s1; s1 = t; }
    const int32_t it = preSi >= 0 ? preSi : allocInteraction();
    Interaction& I = inters[size_t(it)];
    I = Interaction();
    I.alive = true;
    I.type = eOVERLAP;
    I.iflags = IFlag::eRB_ELEMENT | IFlag::eFILTERABLE;
    I.elem0 = s0;
    I.elem1 = s1;
    I.actor0 = shapes[size_t(s0)].actor;
    I.actor1 = shapes[size_t(s1)].actor;
    elementSimMap[key(s0, s1)] = it;  // ElementSimInteraction 생성자 (NPhaseCore::registerInteraction)
    // ShapeInteraction 생성자 (ScShapeInteraction.cpp:38)
    setPairFlags(I, pairFlags);
    updateFlags(I, pairFlags);
    if (preCm < 0) {
      Actor& b0 = actors[size_t(I.actor0)];
      Actor& b1 = actors[size_t(I.actor1)];
      const uint64_t index0 = b0.nodeIndex;
      b0.countedInteractions++;
      uint64_t index1 = INVALID_NODE;
      if (!b1.isStatic()) {
        index1 = b1.nodeIndex;
        b1.countedInteractions++;
      }
      I.edge = islands->addContactManager(-1, index0, index1, it, eEDGE_CONTACT_MANAGER);
      const bool active = onActivate(it, -1);
      (void)active;
      registerInActor(I.actor0, it, 0);
      registerInActor(I.actor1, it, 1);
      I.sceneRegistered = true;
    } else {
      onActivate(it, preCm);
    }
    return it;
  }
  SCHD int32_t createTriggerInteraction(int32_t s0, int32_t s1, uint32_t triggerFlags) {  // :277 (트리거 모양이 앞)
    int32_t trig = s0, other = s1;
    if (shapes[size_t(s1)].trigger) { trig = s1; other = s0; }
    const int32_t it = allocInteraction();
    Interaction& I = inters[size_t(it)];
    I.alive = true;
    I.type = eTRIGGER;
    I.iflags = IFlag::eRB_ELEMENT | IFlag::eFILTERABLE;
    I.elem0 = trig;
    I.elem1 = other;
    I.actor0 = shapes[size_t(trig)].actor;
    I.actor1 = shapes[size_t(other)].actor;
    I.siFlags = triggerFlags;
    elementSimMap[key(trig, other)] = it;
    // TriggerInteraction 생성자: registerInActors + 장면 등록 (활성은 트리거 처리 몫 — 아직)
    registerInActor(I.actor0, it, 0);
    registerInActor(I.actor1, it, 1);
    I.sceneRegistered = true;
    return it;
  }
  SCHD int32_t createMarker(int32_t e0, int32_t e1, bool createParallel) {  // :297, ScElementInteractionMarker.h:171
    const int32_t it = allocInteraction();
    Interaction& I = inters[size_t(it)];
    I.alive = true;
    I.type = eMARKER;
    I.iflags = IFlag::eRB_ELEMENT | IFlag::eFILTERABLE;
    I.elem0 = e0;
    I.elem1 = e1;
    I.actor0 = shapes[size_t(e0)].actor;
    I.actor1 = shapes[size_t(e1)].actor;
    elementSimMap[key(e0, e1)] = it;
    if (!createParallel) {
      registerInActor(I.actor0, it, 0);
      registerInActor(I.actor1, it, 1);
      I.sceneRegistered = true;
    }
    return it;
  }
  SCHD static void setPairFlags(Interaction& I, uint32_t flags) {  // ScShapeInteraction.h:208
    I.siFlags = (I.siFlags & ~SiFlag::PAIR_FLAGS_MASK) | (flags & SiFlag::PAIR_FLAGS_MASK);
  }
  SCHD void updateFlags(Interaction& I, uint32_t pairFlags) {  // ScShapeInteraction.cpp:747
    const Actor& b0 = actors[size_t(I.actor0)];
    const Actor& b1 = actors[size_t(I.actor1)];
    bool enabled = true;
    if (b0.isDynamicRigid()) enabled = !b0.isKinematic();
    if (b1.isDynamicRigid()) enabled |= !b1.isKinematic();
    enabled = enabled && (pairFlags & PairFlag::eSOLVE_CONTACT);
    if (!enabled) I.siFlags |= SiFlag::CONTACTS_RESPONSE_DISABLED;
    else I.siFlags &= ~SiFlag::CONTACTS_RESPONSE_DISABLED;
    const bool collect = (pairFlags & PairFlag::eNOTIFY_CONTACT_POINTS) || (pairFlags & PairFlag::eMODIFY_CONTACTS);  // 시각화 끔
    if (collect) I.siFlags |= SiFlag::CONTACTS_COLLECT_POINTS;
    else I.siFlags &= ~SiFlag::CONTACTS_COLLECT_POINTS;
  }
  SCHD bool activeManagerAllowed(const Interaction& I) {  // ScShapeInteraction.h:286
    const Actor& b0 = actors[size_t(I.actor0)];
    const Actor& b1 = actors[size_t(I.actor1)];
    return islands->isSpeculativeNodeActive(b0.nodeIndex) || (!b1.isStatic() && islands->isSpeculativeNodeActive(b1.nodeIndex));
  }
  SCHD bool onActivate(int32_t it, int32_t preCm) {  // ScShapeInteraction.cpp:921 (+ updateManager, ScShapeInteraction.h:257)
    Interaction& I = inters[size_t(it)];
    bool ok = false;
    if (activeManagerAllowed(I)) {
      if (I.cm < 0) createManager(it, preCm);
      ok = I.cm >= 0;
    }
    if (ok) I.iflags |= IFlag::eIS_ACTIVE;
    return ok;
  }
  SCHD bool onDeactivate(int32_t it) {  // ScShapeInteraction.cpp:939
    Interaction& I = inters[size_t(it)];
    if (I.type != eOVERLAP) return false;
    const Actor& b0 = actors[size_t(I.actor0)];
    const Actor& b1 = actors[size_t(I.actor1)];
    if (!islands->isActorActive(I.actor0) && (b1.isStatic() || !islands->isActorActive(I.actor1))) {
      if (I.cm >= 0) {
        const ContactManager& cm = cmsData[size_t(I.cm)];
        const bool touchKnown = (cm.statusFlags & (WuStatus::eHAS_TOUCH | WuStatus::eHAS_NO_TOUCH)) != 0;
        const bool touch = (cm.statusFlags & WuStatus::eHAS_TOUCH) != 0;
        if (!(I.siFlags & SiFlag::TOUCH_KNOWN) && touchKnown && !touch) I.siFlags |= SiFlag::HAS_NO_TOUCH;
        destroyManager(it);
        if (I.edge != INVALID) islands->clearEdgeRigidCM(I.edge);
      }
      islands->deactivateEdge(I.edge);
      I.iflags &= uint8_t(~IFlag::eIS_ACTIVE);
      return true;
    }
    return false;
  }
  SCHD void createManager(int32_t it, int32_t preCm) {  // ScShapeInteraction.cpp:1004
    Interaction& I = inters[size_t(it)];
    const uint32_t pairFlags = I.siFlags & SiFlag::PAIR_FLAGS_MASK;
    const bool disableCCDContact = !(pairFlags & PairFlag::eDETECT_CCD_CONTACT);
    // PxsContext::createContactManager (PxsContext.cpp:264): 풀에서 (미리 뽑은 것이 없으면) 하나
    const int32_t cmi = preCm >= 0 ? preCm : cmPool.get();
    if (cmsData.size() < cmPool.capacity()) cmsData.resize(cmPool.capacity());
    ContactManager& cm = cmsData[size_t(cmi)];
    cm.cachedStateEpoch++;  // clearCachedState
    const bool contactChangeable = (pairFlags & PairFlag::eMODIFY_CONTACTS) != 0;
    const Shape& sh0 = shapes[size_t(I.elem0)];
    const Shape& sh1 = shapes[size_t(I.elem1)];
    const Actor& b0 = actors[size_t(I.actor0)];
    const Actor& b1 = actors[size_t(I.actor1)];
    const bool disableResponse = (I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED) != 0;
    const bool disableDiscreteContact = !(pairFlags & PairFlag::eDETECT_DISCRETE_CONTACT);
    const bool reportContactInfo = (I.siFlags & SiFlag::CONTACTS_COLLECT_POINTS) != 0;
    const bool hasForceThreshold = !disableResponse && (pairFlags & SiFlag::CONTACT_FORCE_THRESHOLD_PAIRS);
    int touching;
    if (I.siFlags & SiFlag::TOUCH_KNOWN) touching = (I.siFlags & SiFlag::HAS_TOUCH) ? 1 : -1;
    else touching = 0;
    const bool kinematicActor = b1.isDynamicRigid() ? b1.isKinematic() : false;
    cm.inter = it;
    cm.shape0 = I.elem0;
    cm.shape1 = I.elem1;
    cm.geomType0 = sh0.geomType;
    cm.geomType1 = sh1.geomType;
    cm.restDistance = sh0.restOffset + sh1.restOffset;
    cm.transformCache0 = sh0.transformCacheId;
    cm.transformCache1 = sh1.transformCacheId;
    cm.torsionalPatchRadius = sh0.torsionalPatchRadius > sh1.torsionalPatchRadius ? sh0.torsionalPatchRadius : sh1.torsionalPatchRadius;  // PxMax
    cm.minTorsionalPatchRadius = sh0.minTorsionalPatchRadius > sh1.minTorsionalPatchRadius ? sh0.minTorsionalPatchRadius : sh1.minTorsionalPatchRadius;
    const float slop0 = b0.isDynamicRigid() ? b0.offsetSlop : 0.0f;
    const float slop1 = b1.isDynamicRigid() ? b1.offsetSlop : 0.0f;
    cm.offsetSlop = slop0 > slop1 ? slop0 : slop1;
    uint16_t wu = 0;
    if (b0.type == eARTICULATION_LINK) wu |= WuFlag::eARTICULATION_BODY0;
    if (b1.type == eARTICULATION_LINK) wu |= WuFlag::eARTICULATION_BODY1;
    if (b0.type == eRIGID_DYNAMIC) wu |= WuFlag::eDYNAMIC_BODY0;
    if (b1.type == eRIGID_DYNAMIC) wu |= WuFlag::eDYNAMIC_BODY1;
    if (!disableResponse && !contactChangeable) wu |= WuFlag::eOUTPUT_CONSTRAINTS;
    if (!disableDiscreteContact) wu |= WuFlag::eDETECT_DISCRETE_CONTACT;
    if (kinematicActor) wu |= WuFlag::eHAS_KINEMATIC_ACTOR;
    if (disableResponse) wu |= WuFlag::eDISABLE_RESPONSE;
    if (!disableCCDContact) wu |= WuFlag::eDETECT_CCD_CONTACTS;
    if (reportContactInfo || contactChangeable) wu |= WuFlag::eOUTPUT_CONTACTS;
    if (hasForceThreshold) wu |= WuFlag::eFORCE_THRESHOLD;
    if (contactChangeable) wu |= WuFlag::eMODIFIABLE_CONTACT;
    cm.wuFlags = wu;
    // setupDominance (ScShapeInteraction.cpp:792)
    const uint8_t dom0 = b0.dominanceGroup;
    const uint8_t dom1 = b1.isStatic() ? 0 : b1.dominanceGroup;
    cm.dominance0 = dominance[dom0][dom1][0];
    cm.dominance1 = dominance[dom0][dom1][1];
    cm.cmFlags = (contactChangeable ? 1u : 0u) | (disableCCDContact ? 0u : 2u);
    cm.npIndex = INVALID;
    uint8_t status = 0;
    if (touching > 0) status |= WuStatus::eHAS_TOUCH;
    else if (touching < 0) status |= WuStatus::eHAS_NO_TOUCH;
    cm.statusFlags = status;
    cm.inUse = true;
    I.cm = cmi;
    if (preCm < 0) {
      islands->setEdgeRigidCM(I.edge, cmi);
      registerContactManager(cmi, touching, 0);
    }
  }
  // PxsNphaseImplementationContext.cpp:643 registerContactManager
  SCHD void registerContactManager(int32_t cmi, int touching, uint32_t patchCount) {
    ContactManager& cm = cmsData[size_t(cmi)];
    NpOutput out;
    out.nbPatches = uint8_t(patchCount);
    if (cm.wuFlags & WuFlag::eOUTPUT_CONSTRAINTS) out.statusFlag |= OutStatus::eREQUEST_CONSTRAINTS;
    if (touching > 0) out.statusFlag |= OutStatus::eHAS_TOUCH;
    else if (touching < 0) out.statusFlag |= OutStatus::eHAS_NO_TOUCH;
    out.statusFlag |= OutStatus::eDIRTY_MANAGER;
    if (cm.statusFlags & WuStatus::eHAS_TOUCH) cm.statusFlags |= WuStatus::eREFRESHED_WITH_TOUCH;
    out.flags = uint8_t(cm.wuFlags);
    const uint32_t slot = npNew.size();
    npNew.cms.push_back(cmi);
    npNew.outputs.push_back(out);
    npNew.cacheKind.push_back(cacheKindFor(cm.geomType0, cm.geomType1));
    if (caches) {
      caches->resize(true, slot + 1);
      caches->create(true, slot, cm.geomType0, cm.geomType1);
    }
    cm.npIndex = (slot << CM_BUCKET_BITS) | NEW_CM_MASK;
  }
  SCHD int32_t cacheKindFor(int32_t g0, int32_t g1) const {  // PxsContext::createCache (PxsContext.cpp:282) 의 종류만
    if (!pcm) return 0;
    static const bool tab[6][6] = {{false, false, false, false, false, true}, {false, false, true, true, false, true}, {false, true, false, true, false, true},
                                   {false, true, true, true, false, true},     {false, false, false, false, false, false}, {true, true, true, true, false, true}};
    if (g0 > 5 || g1 > 5) return 3;  // 메시·높이장: 다중 다양체 (gEnablePCMCaching 표에서 메시 쪽은 모두 참)
    if (!tab[g0][g1]) return 0;
    return (g0 == 0 || g1 == 0) ? 1 : 2;
  }
  // PxsNphaseImplementationContext.cpp:709 unregisterContactManager -> unregisterAndForceSize -> :967 unregisterContactManagerInternal
  SCHD void unregisterContactManager(int32_t cmi) {
    const uint32_t index = cmsData[size_t(cmi)].npIndex;
    const bool isNew = (index & NEW_CM_MASK) != 0;
    NpList& L = isNew ? npNew : npMain;
    const uint32_t slot = (index & ~NEW_CM_MASK) >> CM_BUCKET_BITS;
    const uint32_t replaceIndex = L.size() - 1;
    const int32_t replaceCm = L.cms[replaceIndex];
    if (caches) {
      caches->destroy(isNew, slot);
      caches->move(isNew, slot, isNew, replaceIndex);
      caches->resize(isNew, replaceIndex);
    }
    L.cms[slot] = replaceCm;
    L.cacheKind[slot] = L.cacheKind[replaceIndex];
    L.outputs[slot] = L.outputs[replaceIndex];
    cmsData[size_t(replaceCm)].npIndex = index;
    L.cms.pop_back();
    L.cacheKind.pop_back();
    L.outputs.pop_back();
  }
  // ShapeInteraction::destroyManager (ScShapeInteraction.h:270) -> PxsContext::destroyContactManager (PxsContext.cpp:321)
  SCHD void destroyManager(int32_t it) {
    Interaction& I = inters[size_t(it)];
    const int32_t cmi = I.cm;
    unregisterContactManager(cmi);
    touchEvent.growAndReset(uint32_t(cmi));
    cmsData[size_t(cmi)].inUse = false;
    cmPool.put(cmi);
    I.cm = -1;
  }
  // ShapeInteraction::updateState (ScShapeInteraction.cpp:803)
  SCHD void updateState(int32_t it, uint8_t externalDirtyFlags) {
    Interaction& I = inters[size_t(it)];
    const uint32_t oldContactState = I.siFlags & SiFlag::LL_MANAGER_RECREATE_EVENT;
    const uint8_t dirtyFlags = uint8_t(I.dirty | externalDirtyFlags);
    const uint32_t pairFlags = I.siFlags & SiFlag::PAIR_FLAGS_MASK;
    if (dirtyFlags & (DirtyFlag::eFILTER_STATE | DirtyFlag::eVISUALIZATION)) {
      const bool wasDisabled = (I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED) != 0;
      updateFlags(I, pairFlags);
      const bool isDisabled = (I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED) != 0;
      if (!wasDisabled && isDisabled) islands->setEdgeDisconnected(I.edge);
      else if (wasDisabled && !isDisabled) {
        if (I.siFlags & SiFlag::HAS_TOUCH) islands->setEdgeConnected(I.edge, eEDGE_CONTACT_MANAGER);
      }
    }
    const uint32_t newContactState = I.siFlags & SiFlag::LL_MANAGER_RECREATE_EVENT;
    const bool recreateManager = oldContactState != newContactState;
    if (!recreateManager && I.cm >= 0) {
      ContactManager& cm = cmsData[size_t(I.cm)];
      if (dirtyFlags & DirtyFlag::eDOMINANCE) {
        const Actor& b0 = actors[size_t(I.actor0)];
        const Actor& b1 = actors[size_t(I.actor1)];
        const uint8_t dom0 = b0.dominanceGroup, dom1 = b1.isStatic() ? 0 : b1.dominanceGroup;
        cm.dominance0 = dominance[dom0][dom1][0];
        cm.dominance1 = dominance[dom0][dom1][1];
      }
      if ((dirtyFlags & DirtyFlag::eBODY_KINEMATIC) == DirtyFlag::eBODY_KINEMATIC) {
        const Actor& b1 = actors[size_t(I.actor1)];
        if (b1.isDynamicRigid()) {
          if (b1.isKinematic()) cm.wuFlags |= WuFlag::eHAS_KINEMATIC_ACTOR;
          else cm.wuFlags &= uint16_t(~WuFlag::eHAS_KINEMATIC_ACTOR);
        }
      }
      if (dirtyFlags & DirtyFlag::eREST_OFFSET) cm.restDistance = shapes[size_t(I.elem0)].restOffset + shapes[size_t(I.elem1)].restOffset;
      cm.cmFlags = (cm.cmFlags & ~2u) | (((I.siFlags & PairFlag::eDETECT_CCD_CONTACT) != 0) ? 2u : 0u);  // setCCD
      if (dirtyFlags) resetManagerCachedState(it);
    } else if (I.iflags & IFlag::eIS_ACTIVE) {
      if ((dirtyFlags & DirtyFlag::eBODY_KINEMATIC) == DirtyFlag::eBODY_KINEMATIC) {
        const Actor& b0 = actors[size_t(I.actor0)];
        const Actor& b1 = actors[size_t(I.actor1)];
        if (!islands->isSpeculativeNodeActiveOrActivating(b0.nodeIndex) && (b1.isStatic() || !islands->isSpeculativeNodeActiveOrActivating(b1.nodeIndex))) {
          onDeactivate(it);
          // scene.notifyInteractionDeactivated (장면 활성 목록 — 순서 영향 없음)
        } else {
          if (I.edge != INVALID) islands->clearEdgeRigidCM(I.edge);
          destroyManager(it);
          createManager(it, -1);
        }
      } else {
        if (I.edge != INVALID) islands->clearEdgeRigidCM(I.edge);
        destroyManager(it);
        createManager(it, -1);
      }
    }
  }
  // ScShapeInteraction.cpp:194 resetManagerCachedState -> PxsNphaseImplementationContext.cpp:727 refreshContactManager
  SCHD void resetManagerCachedState(int32_t it) {
    Interaction& I = inters[size_t(it)];
    if (I.cm < 0) return;
    ContactManager& cm = cmsData[size_t(I.cm)];
    cm.cachedStateEpoch++;  // clearCachedState (마찰 패치 0)
    const uint32_t index = cm.npIndex;
    const bool isNew = (index & NEW_CM_MASK) != 0;
    NpList& L = isNew ? npNew : npMain;
    const NpOutput output = L.outputs[(index & ~NEW_CM_MASK) >> CM_BUCKET_BITS];
    unregisterContactManager(I.cm);
    int touching = 0;
    if (output.statusFlag & OutStatus::eHAS_TOUCH) touching = 1;
    else if (output.statusFlag & OutStatus::eHAS_NO_TOUCH) touching = -1;
    registerContactManager(I.cm, touching, output.nbPatches);
  }
  // ShapeInteraction::managerLostTouch (ScShapeInteraction.cpp:703), 보고 없음
  SCHD bool managerLostTouch(int32_t it) {
    Interaction& I = inters[size_t(it)];
    if (!(I.siFlags & SiFlag::HAS_TOUCH)) return false;
    I.siFlags &= ~SiFlag::HAS_TOUCH;
    I.siFlags |= SiFlag::HAS_NO_TOUCH;
    if (actors[size_t(I.actor1)].isStatic()) {
      islands->internalWakeUp(I.actor0);
      return false;
    }
    return true;
  }
  SCHD void clearIslandGenData(int32_t it) {  // ScShapeInteraction.cpp:150
    Interaction& I = inters[size_t(it)];
    if (I.edge != INVALID) {
      islands->removeConnection(I.edge);
      I.edge = INVALID;
    }
  }
  // NPhaseCore::lostTouchReports (ScNPhaseCore.cpp:1003) 중 깨우기 규칙 (보고·행위자 쌍 셈 없음)
  SCHD void lostTouchReportsOne(int32_t it, bool wakeOnLostTouch, int32_t removedElement) {
    const Interaction& I = inters[size_t(it)];
    const bool hasTouch = (I.siFlags & SiFlag::HAS_TOUCH) != 0;
    // ShapeInteraction::hasKnownTouchState (ScShapeInteraction.h:329): 관리자가 있으면 관리자 작업 단위의 닿음 상태(좁은 단계가 한 번이라도 돌았나),
    // 없을 때만 상호작용 표시 (09-30 리드: 표시만 봐서 아직 한 번도 안 닿은 쌍을 "모름"으로 보고 깨우기 목록에 더 넣었다 — 양파 961)
    const bool touchKnown = I.cm >= 0 ? (cmsData[size_t(I.cm)].statusFlags & (WuStatus::eHAS_TOUCH | WuStatus::eHAS_NO_TOUCH)) != 0
                                      : (I.siFlags & SiFlag::TOUCH_KNOWN) != 0;
    if (hasTouch || !touchKnown) {
      if (wakeOnLostTouch) {
        const Actor& b1 = actors[size_t(I.actor1)];
        if (removedElement < 0) {
          if (b1.isStatic()) islands->internalWakeUp(I.actor0);
          else if (!(I.siFlags & SiFlag::CONTACTS_RESPONSE_DISABLED)) islands->addToLostTouchList(I.actor0, I.actor1);
        } else {
          if (I.elem0 == removedElement) {
            if (!b1.isStatic()) islands->internalWakeUp(I.actor1);
          } else {
            islands->internalWakeUp(I.actor0);
          }
        }
      }
    }
  }
  // NPhaseCore::onOverlapRemoved (ScNPhaseCore.cpp:89) -> releaseElementPair (:946)
  SCHD void onOverlapRemoved(int32_t e0, int32_t e1, int32_t knownInter) {
    const int32_t it = knownInter >= 0 ? knownInter : findInteraction(e1, e0);
    if (it < 0) return;
    releaseElementPair(it, PairRelease::eWAKE_ON_LOST_TOUCH, -1);
  }
  SCHD void releaseElementPair(int32_t it, uint32_t flags, int32_t removedElement, bool removeFromDirtyList = true) {
    Interaction& I = inters[size_t(it)];
    // setClean(removeFromDirtyList) (ScInteraction.cpp:59)
    if (I.iflags & IFlag::eIN_DIRTY_LIST) {
      if (removeFromDirtyList) {
        const uint32_t pos = inters[size_t(it)].dirtyPos;
        const int32_t last = dirtyList.back();
        dirtyList[pos] = last;
        inters[size_t(last)].dirtyPos = pos;
        dirtyList.pop_back();
        inters[size_t(it)].dirtyPos = INVALID;
      }
      I.iflags &= uint8_t(~IFlag::eIN_DIRTY_LIST);
    }
    I.dirty = 0;
    if ((I.iflags & IFlag::eIS_FILTER_PAIR) && filterPairLost) {
      filterPairLost(pairID(I.elem0, I.elem1), filterAttrOf(I.elem0, true), shapes[size_t(I.elem0)].fd, filterAttrOf(I.elem1, true), shapes[size_t(I.elem1)].fd,
                     removedElement >= 0, filterCallbackData);
    }
    if (I.type == eOVERLAP) {
      if (flags & PairRelease::eRUN_LOST_TOUCH_LOGIC) lostTouchReportsOne(it, (flags & PairRelease::eWAKE_ON_LOST_TOUCH) != 0, removedElement);
      // ~ShapeInteraction (ScShapeInteraction.cpp:120)
      Actor& b0 = actors[size_t(I.actor0)];
      Actor& b1 = actors[size_t(I.actor1)];
      if (b0.countedInteractions) b0.countedInteractions--;  // unregisterCountedInteraction (동적만 셈)
      if (b1.isDynamicRigid() && b1.countedInteractions) b1.countedInteractions--;
      if (I.cm >= 0) destroyManager(it);
      if (I.edge != INVALID) {
        islands->removeConnection(I.edge);
        I.edge = INVALID;
        I.sceneRegistered = false;
      }
      unregisterFromActors(it);
    } else if (I.type == eMARKER) {
      I.sceneRegistered = false;
      unregisterFromActors(it);
    } else if (I.type == eTRIGGER) {
      I.sceneRegistered = false;
      unregisterFromActors(it);
    }
    elementSimMap.erase(key(I.elem0, I.elem1));  // ~ElementSimInteraction -> NPhaseCore::unregisterInteraction
    freeInteraction(it);
  }
};

}  // namespace sc
}  // namespace contact
}  // namespace eng
