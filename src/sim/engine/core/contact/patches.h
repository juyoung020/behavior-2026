// 접촉 패치 만들기 (호스트·GPU 공용, 손으로 옮김): 좁은 단계 접촉점 목록 -> PhysX 압축 접촉 흐름(패치 머리 + 접촉점).
// 원본: lowlevel/common/src/pipeline/PxcNpContactPrepShared.cpp:74 writeCompressedContact (5.6.1),
//       재질 섞기 lowlevel/software/include/PxsMaterialCombiner.h:88 PxsCombineMaterials,
//       재질 번호 lowlevel/common/src/pipeline/PxcMaterialMethodImpl.cpp (모양-모양: 두 모양의 재질 번호를 모든 점에).
// 풀이(solver)가 읽는 배치 그대로: PxContactPatch 64 바이트(질량 배율 16 + 법선 16 + 재질값·번호 32), PxContact 16 바이트.
// 수정 가능 접촉(eMODIFY_CONTACTS, omni 표면 속도 전용)은 안 옮겼다 -> hasModifiableContacts 는 늘 거짓으로 둔다.
#pragma once
#include <cstdint>

#include "core/contact/px/gu.h"

namespace eng {
namespace contact {

// include/PxContact.h:57 (PX_ALIGN 16)
struct alignas(16) ContactPatch {
  enum Flags : uint16_t { eHAS_FACE_INDICES = 1, eMODIFIABLE = 2, eFORCE_NO_RESPONSE = 4 };
  float linear0, angular0, linear1, angular1;  // PxConstraintInvMassScale (PxConstraintDesc.h:245)
  px::PxVec3 normal;                           // 오프셋 16 (PX_ALIGN 16), restitution 은 바로 뒤 28
  float restitution;
  float dynamicFriction;
  float staticFriction;
  float damping;
  uint16_t startContactIndex;
  uint8_t nbContacts;
  uint8_t materialFlags;
  uint16_t internalFlags;
  uint16_t materialIndex0;
  uint16_t materialIndex1;
  uint16_t pad[5];
};
static_assert(sizeof(ContactPatch) == 64, "PxContactPatch 64 바이트");

struct alignas(16) Contact {  // include/PxContact.h:140
  px::PxVec3 contact;
  float separation;
};
static_assert(sizeof(Contact) == 16, "PxContact 16 바이트");

// lowlevel/api/include/PxsMaterialCore.h:40 PxsMaterialData
struct MaterialData {
  float dynamicFriction = 0.0f;
  float staticFriction = 0.0f;
  float restitution = 0.0f;
  float damping = 0.0f;
  uint16_t flags = 0;  // PxMaterialFlags
  uint8_t fricCombineMode = 0, restCombineMode = 0, dampingCombineMode = 0;  // PxCombineMode (0 평균, 1 최소, 2 곱, 3 최대)
};
enum MaterialFlag : uint32_t {  // include/PxMaterial.h
  eDISABLE_FRICTION = 1 << 0,
  eDISABLE_STRONG_FRICTION = 1 << 1,
  eCOMPLIANT_ACCELERATION_SPRING = 1 << 4
};
enum CombineMode : int32_t { eAVERAGE = 0, eMIN = 1, eMULTIPLY = 2, eMAX = 3 };

EHD float combineScalars(float a, float b, int32_t mode) {  // PxsMaterialCombiner.h:37
  switch (mode) {
    case eAVERAGE: return 0.5f * (a + b);
    case eMIN: return px::PxMin(a, b);
    case eMULTIPLY: return a * b;
    case eMAX: return px::PxMax(a, b);
    default: return 0.0f;
  }
}
EHD float combinePxReal(float a, float b, int32_t mode) {  // PxsMaterialCombiner.h:56
  switch (mode) {
    case eAVERAGE: return 0.5f * (a + b);
    case eMIN: return px::PxMin(a, b);
    case eMULTIPLY: return (a * b);
    case eMAX: return px::PxMax(a, b);
  }
  return 0.0f;
}

// PxsMaterialCombiner.h:73 PxsCombineMaterials (CPU 판: fsel 쪽)
EHD void combineMaterials(const MaterialData& m0, const MaterialData& m1, float& sf, float& df, float& rest, uint32_t& flags, float& damp) {
  const float r0 = m0.restitution, r1 = m1.restitution;
  const bool compliant0 = r0 < 0.0f, compliant1 = r1 < 0.0f;
  const bool exactlyOneCompliant = compliant0 ^ compliant1;
  const bool bothCompliant = compliant0 & compliant1;
  const bool acc0 = !!(m0.flags & eCOMPLIANT_ACCELERATION_SPRING), acc1 = !!(m1.flags & eCOMPLIANT_ACCELERATION_SPRING);
  const bool exactlyOneAcc = acc0 ^ acc1;
  if (bothCompliant && exactlyOneAcc) {
    rest = acc0 ? r0 : r1;
  } else {
    const int32_t mode = exactlyOneCompliant ? int32_t(eMIN) : px::PxMax(int32_t(m0.restCombineMode), int32_t(m1.restCombineMode));
    const float flipSign = (bothCompliant && (mode == eMULTIPLY)) ? -1.0f : 1.0f;
    rest = flipSign * combineScalars(r0, r1, mode);
  }
  {
    const float d0 = m0.damping, d1 = m1.damping;
    if (bothCompliant && exactlyOneAcc) {
      damp = acc0 ? d0 : d1;
    } else {
      const int32_t mode = exactlyOneCompliant ? int32_t(eMAX) : px::PxMax(int32_t(m0.dampingCombineMode), int32_t(m1.dampingCombineMode));
      damp = combineScalars(d0, d1, mode);
    }
  }
  const uint32_t combineFlags = uint32_t(m0.flags | m1.flags);
  if (!(combineFlags & eDISABLE_FRICTION)) {
    const int32_t mode = px::PxMax(int32_t(m0.fricCombineMode), int32_t(m1.fricCombineMode));
    const float dyn = combinePxReal(m0.dynamicFriction, m1.dynamicFriction, mode);
    const float sta = combinePxReal(m0.staticFriction, m1.staticFriction, mode);
    const float fDyn = px::PxMax(dyn, 0.0f);
    const float fSta = px::intrinsics::fsel(sta - fDyn, sta, fDyn);
    df = fDyn;
    sf = fSta;
    flags = combineFlags;
  } else {
    flags = combineFlags | eDISABLE_STRONG_FRICTION;
    df = 0.0f;
    sf = 0.0f;
  }
}

// 한 쌍의 압축 접촉 출력 (PxsContactManagerOutput 의 패치·점 부분). 용량을 넘으면 overflow.
template <int MaxPatches, int MaxContacts>
struct CompressedContacts {
  ContactPatch patches[MaxPatches];
  Contact contacts[MaxContacts];
  uint32_t faceIndices[MaxContacts];  // 삼각메시 등(isMeshType)일 때만 채움
  uint16_t nbContacts;
  uint8_t nbPatches;
  uint8_t overflow;
};

struct MaterialPair {
  uint16_t m0, m1;
};

// PxcNpContactPrepShared.cpp:74 writeCompressedContact (수정 불가 경로). 반환 = 쓴 접촉 수 (평균점 포함).
// mat[i] = 점 i 의 재질 번호 쌍 (모양-모양이면 전부 같음), materials = 재질 표.
template <int MaxPatches, int MaxContacts>
EHD uint32_t writeCompressedContact(const px::PxContactPoint* contactPoints, uint32_t numContactPoints, const MaterialPair* mat,
                                    const MaterialData* materials, bool insertAveragePoint, bool isMeshType,
                                    CompressedContacts<MaxPatches, MaxContacts>& out) {
  out.nbContacts = 0;
  out.nbPatches = 0;
  out.overflow = 0;
  if (numContactPoints == 0) return 0;

  struct StridePatch {
    uint8_t startIndex, endIndex, nextIndex, totalCount;
    bool isRoot;
  };
  StridePatch stridePatches[256];  // PX_ALLOCA(numContactPoints) — PxContactBuffer 최대 256

  uint32_t numStrideHeaders = 1;
  uint32_t totalUniquePatches = 1;
  uint32_t totalContactPoints = numContactPoints;
  uint32_t strideStart = 0;
  bool root = true;
  StridePatch* parentRootPatch = nullptr;
  {
    const float closeNormalThresh = 0.999f;  // PXC_SAME_NORMAL (PxcNpContactPrepShared.h:49)
    px::PxVec3 normal = contactPoints[0].normal;
    uint16_t mat0 = mat[0].m0, mat1 = mat[0].m1;
    for (uint32_t a = 1; a < numContactPoints; ++a) {
      if (normal.dot(contactPoints[a].normal) < closeNormalThresh || mat[a].m0 != mat0 || mat[a].m1 != mat1) {
        StridePatch& patch = stridePatches[numStrideHeaders - 1];
        patch.startIndex = uint8_t(strideStart);
        patch.endIndex = uint8_t(a);
        patch.nextIndex = 0xFF;
        patch.totalCount = uint8_t(a - strideStart);
        patch.isRoot = root;
        if (parentRootPatch) parentRootPatch->totalCount += uint8_t(a - strideStart);
        root = true;
        parentRootPatch = nullptr;
        for (uint32_t b = 1; b < numStrideHeaders; ++b) {
          StridePatch& thisPatch = stridePatches[b - 1];
          if (thisPatch.isRoot) {
            const uint32_t ind = thisPatch.startIndex;
            const float dp2 = contactPoints[a].normal.dot(contactPoints[ind].normal);
            if (dp2 >= closeNormalThresh && mat[a].m0 == mat[ind].m0 && mat[a].m1 == mat[ind].m1) {
              uint32_t nextInd = b - 1;
              while (stridePatches[nextInd].nextIndex != 0xFF) nextInd = stridePatches[nextInd].nextIndex;
              stridePatches[nextInd].nextIndex = uint8_t(numStrideHeaders);
              root = false;
              parentRootPatch = &stridePatches[b - 1];
              break;
            }
          }
        }
        normal = contactPoints[a].normal;
        mat0 = mat[a].m0;
        mat1 = mat[a].m1;
        totalContactPoints = insertAveragePoint && (a - strideStart) > 1 ? totalContactPoints + 1 : totalContactPoints;
        strideStart = a;
        numStrideHeaders++;
        if (root) totalUniquePatches++;
      }
    }
    totalContactPoints = insertAveragePoint && (numContactPoints - strideStart) > 1 ? totalContactPoints + 1 : totalContactPoints;
  }
  {
    StridePatch& patch = stridePatches[numStrideHeaders - 1];
    patch.startIndex = uint8_t(strideStart);
    patch.endIndex = uint8_t(numContactPoints);
    patch.nextIndex = 0xFF;
    patch.totalCount = uint8_t(numContactPoints - strideStart);
    patch.isRoot = root;
    if (parentRootPatch) parentRootPatch->totalCount += uint8_t(numContactPoints - strideStart);
  }
  if (totalUniquePatches > uint32_t(MaxPatches) || totalContactPoints > uint32_t(MaxContacts)) {
    out.overflow = 1;
    return 0;
  }
  out.nbPatches = uint8_t(totalUniquePatches);

  uint16_t origMat0 = mat[0].m0, origMat1 = mat[0].m1;
  float staticFriction, dynamicFriction, combinedRestitution, combinedDamping;
  uint32_t materialFlags;
  combineMaterials(materials[origMat0], materials[origMat1], staticFriction, dynamicFriction, combinedRestitution, materialFlags, combinedDamping);

  const uint32_t flags = isMeshType ? uint32_t(ContactPatch::eHAS_FACE_INDICES) : 0u;
  ContactPatch* patch = out.patches;
  Contact* point = out.contacts;
  uint32_t* faceIndice = isMeshType ? out.faceIndices : nullptr;
  uint32_t currentIndex = 0;
  for (uint32_t a = 0; a < numStrideHeaders; ++a) {
    StridePatch& rootPatch = stridePatches[a];
    if (!rootPatch.isRoot) continue;
    const uint32_t startIndex = rootPatch.startIndex;
    const uint16_t matIndex0 = mat[startIndex].m0, matIndex1 = mat[startIndex].m1;
    if (matIndex0 != origMat0 || matIndex1 != origMat1) {
      combineMaterials(materials[matIndex0], materials[matIndex1], staticFriction, dynamicFriction, combinedRestitution, materialFlags, combinedDamping);
      origMat0 = matIndex0;
      origMat1 = matIndex1;
    }
    // fillPatch (PxcNpContactPrepShared.cpp:334)
    patch->linear0 = 1.0f; patch->linear1 = 1.0f; patch->angular0 = 1.0f; patch->angular1 = 1.0f;
    patch->normal = contactPoints[startIndex].normal;
    patch->restitution = combinedRestitution;
    patch->dynamicFriction = dynamicFriction;
    patch->staticFriction = staticFriction;
    patch->damping = combinedDamping;
    patch->startContactIndex = uint16_t(currentIndex);
    patch->nbContacts = rootPatch.totalCount;
    patch->materialFlags = uint8_t(materialFlags);
    patch->internalFlags = uint16_t(flags);
    patch->materialIndex0 = matIndex0;
    patch->materialIndex1 = matIndex1;
    for (int k = 0; k < 5; ++k) patch->pad[k] = 0;

    if (insertAveragePoint && rootPatch.totalCount > 1) {  // :493
      patch->nbContacts++;
      px::PxVec3 avgPt(0.0f);
      float avgPen(0.0f);
      const float recipCount = px::em_div1(1.0f, float(rootPatch.totalCount));
      uint32_t index = a;
      while (index != 0xFF) {
        StridePatch& p = stridePatches[index];
        for (uint32_t b = p.startIndex; b < p.endIndex; ++b) {
          avgPt += contactPoints[b].point;
          avgPen += contactPoints[b].separation;
        }
        index = stridePatches[index].nextIndex;
      }
      if (faceIndice) {  // 원본은 index == 0xFF 인 칸을 읽는다 (:514, stridePatches[0xFF]) — 삼각메시+평균점 조합에서만. 그 값을 흉내 못 하므로 0
        *faceIndice = 0;
        faceIndice++;
      }
      point->contact = avgPt * recipCount;
      point->separation = avgPen * recipCount;
      point++;
      currentIndex++;
    }
    uint32_t index = a;
    while (index != 0xFF) {
      StridePatch& p = stridePatches[index];
      for (uint32_t b = p.startIndex; b < p.endIndex; ++b) {
        // copyContactPoint (:47): point = cp->point, separation
        point->contact = contactPoints[b].point;
        point->separation = contactPoints[b].separation;
        if (faceIndice) {
          *faceIndice = contactPoints[b].internalFaceIndex1;
          faceIndice++;
        }
        point++;
        currentIndex++;
      }
      index = stridePatches[index].nextIndex;
    }
    patch++;
  }
  out.nbContacts = uint16_t(totalContactPoints);
  return totalContactPoints;
}

}  // namespace contact
}  // namespace eng
