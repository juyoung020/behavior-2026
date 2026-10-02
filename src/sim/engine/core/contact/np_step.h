// 접촉 관리자 하나의 좁은 단계 = PhysX PxcDiscreteNarrowPhasePCM (lowlevel/common/src/pipeline/PxcNpBatch.cpp:260-453) 를 그대로.
//  - checkContactsMustBeGenerated(:261): 감지 끔이면 건너뜀, 더러운 관리자가 아니고 두 몸체 모두 움직이지 않으면(정적·얼림) 지난 출력 유지(copyBuffers)
//  - 접촉 거리 = 두 모양 contactOffset 합(변환 캐시 번호로 찾음, :300)
//  - 모양 번호 순서로 뒤집어 PCM 함수 표를 부르고(narrowphase.h pcmPair), 재질(모양-모양, PxcMaterialMethodImpl.cpp:52) 후 뒤집기 되돌림(:411)
//  - finishContacts(:189): 상태(닿음/안 닿음) + 압축 접촉 스트림(patches.h writeCompressedContact)
// 쌍 관리층(sc_pairs.h)이 좁은 단계 목록의 칸마다 부르고, 결과 상태·패치 수를 ScPairs::narrowPhaseResult 로 넘긴다.
// 호스트·GPU 공용(EHD). 출력 스트림은 칸마다 들고 있다가 건너뛴 스텝에는 그대로 둔다(PhysX 는 새 블록에 복사, 값은 같음).
#pragma once
#include "core/contact/narrowphase.h"
#include "core/contact/patches.h"

namespace eng {
namespace contact {

namespace NpStatus {  // PxsContactManagerStatusFlag
enum : uint8_t { eHAS_NO_TOUCH = 1 << 0, eHAS_TOUCH = 1 << 1, eREQUEST_CONSTRAINTS = 1 << 3, eDIRTY_MANAGER = 1 << 5, eTOUCH_KNOWN = eHAS_NO_TOUCH | eHAS_TOUCH };
}
namespace NpWu {  // PxcNpWorkUnitFlag (쓰는 것만)
enum : uint16_t {
  eOUTPUT_CONTACTS = 1 << 0, eARTICULATION_BODY0 = 1 << 3, eARTICULATION_BODY1 = 1 << 4, eDYNAMIC_BODY0 = 1 << 5, eDYNAMIC_BODY1 = 1 << 6,
  eSOFT_BODY = 1 << 7, eMODIFIABLE_CONTACT = 1 << 8, eFORCE_THRESHOLD = 1 << 9, eDETECT_DISCRETE_CONTACT = 1 << 10
};
}

// 변환 캐시 한 칸 (PxsCachedTransform: PxTransform + flags, eFROZEN = 1)
struct CachedTransform {
  px::PxTransform32 transform;
  uint32_t flags;
};

// 좁은 단계 모양 자료 (요소 번호 = 변환 캐시 번호로 찾음)
struct NpShape {
  ShapeGeom geom;
  uint16_t material;
};

struct NpParams {
  float meshContactMargin = 0.01f;  // PxcNpThreadContext::mNarrowPhaseParams (PxTolerancesScale 에서)
  float toleranceLength = 1.0f;
  bool createAveragePoint = false;  // PxSceneFlag::eENABLE_AVERAGE_POINT
};

// 칸 하나의 출력 (PxsContactManagerOutput 의 값 부분 + 압축 스트림)
template <int MaxPatches = 32, int MaxContacts = 256>
struct NpSlotOutput {
  uint8_t statusFlag = 0;
  uint8_t nbPatches = 0;
  uint16_t nbContacts = 0;
  CompressedContacts<MaxPatches, MaxContacts> stream;
};

// 작업 단위 입력 (sc_pairs.h ContactManager 와 같은 값)
struct NpWorkUnit {
  uint16_t flags;
  int32_t shape0, shape1;  // 요소 번호 (= 변환 캐시 번호)
};

// PxcDiscreteNarrowPhasePCM. statusFlag 는 호출 전 값(더러움 표시 포함)을 받아 바꾼다. 반환 = 좁은 단계를 실제로 돌렸는지.
template <int MaxPatches, int MaxContacts>
EHD bool discreteNarrowPhasePCM(const NpWorkUnit& wu, const NpShape* shapes, const CachedTransform* tc, const float* contactDistances,
                                const MaterialData* materials, const NpParams& params, ManifoldSlot& cache, px::PxContactBuffer& buf,
                                NpSlotOutput<MaxPatches, MaxContacts>& out) {
  const NpShape& s0 = shapes[wu.shape0];
  const NpShape& s1 = shapes[wu.shape1];
  const CachedTransform& ct0 = tc[wu.shape0];
  const CachedTransform& ct1 = tc[wu.shape1];
  // checkContactsMustBeGenerated (:261)
  if (!(wu.flags & NpWu::eDETECT_DISCRETE_CONTACT)) return false;
  if (!(out.statusFlag & NpStatus::eDIRTY_MANAGER) && !(wu.flags & NpWu::eMODIFIABLE_CONTACT)) {
    const uint32_t body0Dynamic = wu.flags & (NpWu::eDYNAMIC_BODY0 | NpWu::eARTICULATION_BODY0 | NpWu::eSOFT_BODY);
    const uint32_t body1Dynamic = wu.flags & (NpWu::eDYNAMIC_BODY1 | NpWu::eARTICULATION_BODY1 | NpWu::eSOFT_BODY);
    const uint32_t active0 = (body0Dynamic && !(ct0.flags & 1u)) ? 1u : 0u;
    const uint32_t active1 = (body1Dynamic && !(ct1.flags & 1u)) ? 1u : 0u;
    if (!(active0 || active1)) return false;  // copyBuffers: 지난 출력 그대로 (다중 다양체 캐시 복사는 메시 전용)
  }
  out.statusFlag &= uint8_t(~NpStatus::eDIRTY_MANAGER);
  const float contactDist = contactDistances[wu.shape0] + contactDistances[wu.shape1];
  // startContacts (:52) — 상태도 0 으로 (등록 때 켠 eREQUEST_CONSTRAINTS 가 여기서 지워진다; 09-30 리드: 빠져 있어 상태 바이트가 PhysX 와 달랐음)
  out.nbContacts = 0;
  out.nbPatches = 0;
  out.statusFlag = 0;
  // 접촉 함수 (뒤집기·법선 반전 포함)
  bool flipped;
  pcmPair(s0.geom, s1.geom, ct0.transform, ct1.transform, contactDist, params.meshContactMargin, params.toleranceLength, cache, buf, flipped);
  // 재질: PxcGetMaterialShapeShape 는 모든 점에 (모양0 재질, 모양1 재질), 뒤집었으면 flipContacts 가 되돌린다 -> 원래 순서
  MaterialPair mp[256];
  for (px::PxU32 k = 0; k < buf.count; ++k) mp[k] = MaterialPair{s0.material, s1.material};
  // finishContacts (:189)
  uint8_t statusFlags = uint8_t(out.statusFlag & ~NpStatus::eTOUCH_KNOWN);
  statusFlags |= buf.count ? NpStatus::eHAS_TOUCH : NpStatus::eHAS_NO_TOUCH;
  if (!buf.count) {
    out.statusFlag = statusFlags;
    out.nbContacts = 0;
    out.nbPatches = 0;
    out.stream.nbContacts = 0;
    out.stream.nbPatches = 0;
    return true;
  }
  out.statusFlag = statusFlags;
  const int t0 = s0.geom.type, t1 = s1.geom.type;
  const int tmax = t0 > t1 ? t0 : t1;
  const bool isMeshType = tmax > eCONVEXMESH;
  writeCompressedContact(buf.contacts, buf.count, mp, materials, params.createAveragePoint, isMeshType, out.stream);
  out.nbContacts = out.stream.nbContacts;
  out.nbPatches = out.stream.nbPatches;
  if (!out.nbContacts) {  // 넘침
    out.statusFlag = uint8_t((out.statusFlag & ~NpStatus::eTOUCH_KNOWN) | NpStatus::eHAS_NO_TOUCH);
    out.nbPatches = 0;
  }
  return true;
}

}  // namespace contact
}  // namespace eng
