// 모양(geometry) 형: PhysX 공개 API 의 PxGeometry·PxSphereGeometry·PxCapsuleGeometry·PxBoxGeometry·PxPlaneGeometry·
// PxConvexMeshGeometry 에서 접촉 코드가 읽는 칸만 같은 이름으로 둔다 (include/geometry/Px*Geometry.h, 5.6.1).
// 차이: 볼록 메시는 PxConvexMesh* 대신 구운 볼록 데이터(Gu::ConvexHullData) 를 직접 가리킨다 (_getHullData 가 같은 것을 돌려줌,
// 원본 GuConvexMesh.h:178). 모양 번호 순서는 원본 PxGeometryType 그대로(접촉 함수 표·뒤집기 규칙이 이 순서에 기댄다).
namespace px {

struct PxGeometryType {
  enum Enum {
    eSPHERE,
    ePLANE,
    eCAPSULE,
    eBOX,
    eCONVEXCORE,
    eCONVEXMESH,
    ePARTICLESYSTEM,
    eTETRAHEDRONMESH,
    eTRIANGLEMESH,
    eHEIGHTFIELD,
    eCUSTOM,
    eGEOMETRY_COUNT,
    eINVALID = -1
  };
};

class PxGeometry {
 public:
  EHD PxGeometryType::Enum getType() const { return mType; }
 protected:
  EHD PxGeometry(PxGeometryType::Enum type) : mType(type) {}
  PxGeometryType::Enum mType;
 public:
  float mTypePadding = 0.0f;
};

template <class Geom> EHD void checkType(const PxGeometry&) {}

class PxSphereGeometry : public PxGeometry {
 public:
  EHD explicit PxSphereGeometry(PxReal ir = 0.0f) : PxGeometry(PxGeometryType::eSPHERE), radius(ir) {}
  PxReal radius;
};

class PxCapsuleGeometry : public PxGeometry {
 public:
  EHD PxCapsuleGeometry(PxReal radius_ = 0.0f, PxReal halfHeight_ = 0.0f)
      : PxGeometry(PxGeometryType::eCAPSULE), radius(radius_), halfHeight(halfHeight_) {}
  PxReal radius;
  PxReal halfHeight;
};

class PxBoxGeometry : public PxGeometry {
 public:
  EHD PxBoxGeometry(PxReal hx = 0.0f, PxReal hy = 0.0f, PxReal hz = 0.0f)
      : PxGeometry(PxGeometryType::eBOX), halfExtents(hx, hy, hz) {}
  EHD explicit PxBoxGeometry(PxVec3 halfExtents_) : PxGeometry(PxGeometryType::eBOX), halfExtents(halfExtents_) {}
  PxVec3 halfExtents;
};

class PxPlaneGeometry : public PxGeometry {
 public:
  EHD PxPlaneGeometry() : PxGeometry(PxGeometryType::ePLANE) {}
};

namespace Gu {
struct ConvexHullData;
}

class PxConvexMeshGeometry : public PxGeometry {
 public:
  EHD PxConvexMeshGeometry() : PxGeometry(PxGeometryType::eCONVEXMESH), scale(PxMeshScale(1.0f)), hullData(nullptr), meshFlags(1) {}
  EHD PxConvexMeshGeometry(const Gu::ConvexHullData* hull, const PxMeshScale& s, PxU8 flags = 1)
      : PxGeometry(PxGeometryType::eCONVEXMESH), scale(s), hullData(hull), meshFlags(flags) {}
  PxMeshScale scale;
  const Gu::ConvexHullData* hullData;
  PxU8 meshFlags;  // PxConvexMeshGeometryFlag::eTIGHT_BOUNDS = 1
};

namespace Gu {
EHD const ConvexHullData* _getHullData(const PxConvexMeshGeometry& convexGeom) { return convexGeom.hullData; }
}

}  // namespace px
