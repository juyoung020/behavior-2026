// 자체 시험용 장면을 우리 PhysX 빌드로 돌려 OVD 를 남긴다 (공식 평가기 없이 재생기를 검증하려고).
// BEHAVIOR 장면에서 쓰는 것과 같은 종류로 구성: TGS, PCM, omni 충돌 거르개(그룹·쌍), 볼록 메시 동체(묶음 aggregate),
// 정적 상자·삼각 메시, 떠 있는 바닥 관절체(회전·직선 관절, 위치/속도 드라이브, 흉내 관절), 중간에 붙였다 떼는 고정 조인트.
//   ovd_selftest <out_dir> [--threads N] [--substeps N] [--seed S]
//   -> out_dir/selftest.ovd, convex.bin, filters.txt, final_state.bin (마지막 자세, 결정성 확인용)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "PxPhysicsAPI.h"
#include "omnipvd/PxOmniPvd.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdWriter.h"
#include "omni_filter.h"
#include "sidecar.h"

using namespace physx;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ovd_selftest <out_dir> [--threads N] [--substeps N] [--seed S] [--no-ovd]\n"); return 2; }
  std::string out = argv[1];
  int threads = 4, substeps = 600, seed = 7;
  bool ovd = true;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--substeps") && i + 1 < argc) substeps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-ovd")) ovd = false;
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxOmniPvd* opvd = nullptr;
  std::string ovd_path = out + "/selftest.ovd";
  if (ovd) {
    opvd = PxCreateOmniPvd(*fnd);
    if (!opvd) { fprintf(stderr, "PxCreateOmniPvd 실패 (libPVDRuntime_64.so 경로 확인)\n"); return 1; }
    OmniPvdFileWriteStream* fs = opvd->getFileWriteStream();
    fs->setFileName(ovd_path.c_str());
    opvd->getWriter()->setWriteStream(static_cast<OmniPvdWriteStream&>(*fs));
  }
  PxTolerancesScale tol(1.0f, 10.0f);  // omni: PxTolerancesScale(1/metersPerUnit, 10/metersPerUnit)
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol, true, nullptr, opvd);
  if (opvd && !opvd->startSampling()) { fprintf(stderr, "OVD 기록 시작 실패\n"); return 1; }
  PxInitExtensions(*phys, nullptr);

  engine::FilterSpec spec;
  spec.group_pairs.insert(engine::pair_key(7, 7));   // 그룹 7 끼리는 충돌 안 함 (OmniGibson fixed_base_fixed_links 흉내)
  spec.filtered_pairs.insert(engine::pair_key(3, 4));  // 거른 쌍 3-4
  spec.any_contact_report = true;
  engine::write_filters(out + "/filters.txt", spec);
  const engine::FilterSpec* specp = &spec;
  static engine::OmniFilterCallback filterCb;

  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0, 0, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(threads);
  sd.filterShader = engine::OmniFilterShader;
  sd.filterShaderData = &specp;
  sd.filterShaderDataSize = sizeof(specp);
  sd.filterCallback = &filterCb;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_STABILIZATION | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  sd.bounceThresholdVelocity = 0.2f;
  PxScene* scene = phys->createScene(sd);

  PxMaterial* mat = phys->createMaterial(0.8f, 0.6f, 0.0f);
  PxCookingParams cp(tol);
  cp.buildGPUData = false;                // omni.physx/plugins/Setup.cpp:869
  cp.buildTriangleAdjacencies = true;

  // 바닥: 정적 상자 + 경사 삼각 메시
  PxRigidStatic* ground = phys->createRigidStatic(PxTransform(PxVec3(0, 0, -0.5f)));
  PxShape* gs = PxRigidActorExt::createExclusiveShape(*ground, PxBoxGeometry(10, 10, 0.5f), *mat);
  gs->setSimulationFilterData(PxFilterData(0, 0, 0, 0));
  scene->addActor(*ground);
  {
    std::vector<PxVec3> v = {{-1, 1, 0}, {1, 1, 0}, {1, 3, 0.4f}, {-1, 3, 0.4f}, {-1, 3, 0}, {1, 3, 0}};
    std::vector<PxU32> t = {0, 1, 2, 0, 2, 3, 1, 5, 2, 0, 3, 4};
    PxTriangleMeshDesc td;
    td.points.count = PxU32(v.size()); td.points.stride = sizeof(PxVec3); td.points.data = v.data();
    td.triangles.count = PxU32(t.size() / 3); td.triangles.stride = 3 * sizeof(PxU32); td.triangles.data = t.data();
    PxTriangleMesh* tm = PxCreateTriangleMesh(cp, td, phys->getPhysicsInsertionCallback());
    PxRigidStatic* ramp = phys->createRigidStatic(PxTransform(PxVec3(0, 0, 0)));
    PxRigidActorExt::createExclusiveShape(*ramp, PxTriangleMeshGeometry(tm), *mat);
    scene->addActor(*ramp);
  }

  // 볼록 메시 동체들 (묶음)
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<engine::ConvexData> hulls;
  std::vector<PxRigidDynamic*> bodies;
  PxAggregate* agg = phys->createAggregate(16, 16, PxGetAggregateFilterHint(PxAggregateType::eGENERIC, false));
  for (int i = 0; i < 8; ++i) {
    std::vector<PxVec3> pts;
    int np = (i % 2) ? 200 : 40;
    for (int k = 0; k < np; ++k) pts.push_back(PxVec3(0.08f * U(rng), 0.06f * U(rng), 0.05f * U(rng)));
    PxConvexMeshDesc cd;
    cd.points.count = PxU32(pts.size()); cd.points.stride = sizeof(PxVec3); cd.points.data = pts.data();
    cd.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    cd.vertexLimit = (i % 2) ? 64 : 32;
    PxConvexMesh* cm = PxCreateConvexMesh(cp, cd, phys->getPhysicsInsertionCallback());
    engine::ConvexData c;
    c.verts.assign(reinterpret_cast<const float*>(cm->getVertices()),
                   reinterpret_cast<const float*>(cm->getVertices()) + 3 * cm->getNbVertices());
    PxU32 maxIdx = 0;
    for (PxU32 p = 0; p < cm->getNbPolygons(); ++p) {
      PxHullPolygon hp; cm->getPolygonData(p, hp);
      engine::ConvexData::Poly q;
      memcpy(q.plane, hp.mPlane, 16); q.nverts = hp.mNbVerts; q.base = hp.mIndexBase;
      c.polys.push_back(q);
      maxIdx = PxMax(maxIdx, PxU32(hp.mIndexBase + hp.mNbVerts));
    }
    c.indices.assign(cm->getIndexBuffer(), cm->getIndexBuffer() + maxIdx);
    hulls.push_back(c);
    PxRigidDynamic* b = phys->createRigidDynamic(PxTransform(PxVec3(-0.6f + 0.17f * i, 0.3f * (i % 3), 0.3f + 0.12f * i),
                                                             PxQuat(0.3f * i, PxVec3(0, 0, 1))));
    b->setName(("/World/obj_" + std::to_string(i)).c_str());
    PxShape* s = PxRigidActorExt::createExclusiveShape(*b, PxConvexMeshGeometry(cm, PxMeshScale(PxVec3(1.0f, 1.0f + 0.1f * i, 1.0f))), *mat);
    s->setContactOffset(0.02f); s->setRestOffset(0.0f);
    s->setSimulationFilterData(PxFilterData(0, i == 3 ? 3 : 0, (i == 5 || i == 6) ? 7 : 0, 0));
    PxRigidBodyExt::updateMassAndInertia(*b, 300.0f);
    b->setSolverIterationCounts(32, 1);
    agg->addActor(*b);
    bodies.push_back(b);
  }
  scene->addAggregate(*agg);
  engine::write_convex_bin(out + "/convex.bin", hulls);

  // 관절체: 떠 있는 바닥 + 회전(위치 드라이브) + 직선(속도 드라이브) + 회전 두 개(흉내 관절)
  PxArticulationReducedCoordinate* art = phys->createArticulationReducedCoordinate();
  art->setName("/World/robot");
  art->setSolverIterationCounts(32, 1);
  art->setArticulationFlag(PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES, true);
  PxArticulationLink* base = art->createLink(nullptr, PxTransform(PxVec3(0.5f, -0.8f, 0.2f)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.2f, 0.2f, 0.1f), *mat)->setSimulationFilterData(PxFilterData(0, 4, 0, 0));
  PxRigidBodyExt::updateMassAndInertia(*base, 500.0f);
  PxArticulationLink* l1 = art->createLink(base, PxTransform(PxVec3(0.5f, -0.8f, 0.45f)));
  PxRigidActorExt::createExclusiveShape(*l1, PxCapsuleGeometry(0.04f, 0.1f), *mat);
  PxRigidBodyExt::updateMassAndInertia(*l1, 300.0f);
  PxArticulationLink* l2 = art->createLink(l1, PxTransform(PxVec3(0.7f, -0.8f, 0.45f)));
  PxRigidActorExt::createExclusiveShape(*l2, PxBoxGeometry(0.1f, 0.03f, 0.03f), *mat);
  PxRigidBodyExt::updateMassAndInertia(*l2, 300.0f);
  PxArticulationLink* f1 = art->createLink(l2, PxTransform(PxVec3(0.85f, -0.78f, 0.45f)));
  PxRigidActorExt::createExclusiveShape(*f1, PxBoxGeometry(0.03f, 0.01f, 0.02f), *mat)->setSimulationFilterData(PxFilterData(0, 0, 7, 0));
  PxRigidBodyExt::updateMassAndInertia(*f1, 300.0f);
  PxArticulationLink* f2 = art->createLink(l2, PxTransform(PxVec3(0.85f, -0.82f, 0.45f)));
  PxRigidActorExt::createExclusiveShape(*f2, PxBoxGeometry(0.03f, 0.01f, 0.02f), *mat)->setSimulationFilterData(PxFilterData(0, 0, 7, 0));
  PxRigidBodyExt::updateMassAndInertia(*f2, 300.0f);

  auto J = [](PxArticulationLink* l) { return l->getInboundJoint(); };
  J(l1)->setJointType(PxArticulationJointType::eREVOLUTE);
  J(l1)->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
  J(l1)->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-2.0f, 2.0f));
  J(l1)->setDriveParams(PxArticulationAxis::eTWIST, PxArticulationDrive(150.0f, 15.0f, 1e4f, PxArticulationDriveType::eFORCE));
  J(l1)->setParentPose(PxTransform(PxVec3(0, 0, 0.12f), PxQuat(PxHalfPi, PxVec3(0, 1, 0))));
  J(l1)->setChildPose(PxTransform(PxVec3(0, 0, -0.13f), PxQuat(PxHalfPi, PxVec3(0, 1, 0))));
  J(l2)->setJointType(PxArticulationJointType::ePRISMATIC);
  J(l2)->setMotion(PxArticulationAxis::eX, PxArticulationMotion::eLIMITED);
  J(l2)->setLimitParams(PxArticulationAxis::eX, PxArticulationLimit(-0.1f, 0.3f));
  J(l2)->setDriveParams(PxArticulationAxis::eX, PxArticulationDrive(0.0f, 150.0f, 500.0f, PxArticulationDriveType::eFORCE));
  J(l2)->setArmature(PxArticulationAxis::eX, 0.01f);
  for (PxArticulationLink* f : {f1, f2}) {
    J(f)->setJointType(PxArticulationJointType::ePRISMATIC);
    J(f)->setMotion(PxArticulationAxis::eY, PxArticulationMotion::eLIMITED);
    J(f)->setLimitParams(PxArticulationAxis::eY, PxArticulationLimit(-0.03f, 0.03f));
    J(f)->setDriveParams(PxArticulationAxis::eY, PxArticulationDrive(1000.0f, 50.0f, 100.0f));
    J(f)->setFrictionParams(PxArticulationAxis::eY, PxJointFrictionParams(0.1f, 0.05f, 0.01f));
  }
  scene->addArticulation(*art);
  art->createMimicJoint(*J(f1), PxArticulationAxis::eY, *J(f2), PxArticulationAxis::eY, 1.0f, 0.0f);

  // 시간 진행: 120 Hz, 서브스텝마다 목표 갱신. 150 에서 고정 조인트 붙이고 350 에서 뗌
  const float dt = 1.0f / 120.0f;
  PxFixedJoint* fj = nullptr;
  for (int s = 0; s < substeps; ++s) {
    const float t = s * dt;
    J(l1)->setDriveTarget(PxArticulationAxis::eTWIST, 0.8f * sinf(1.3f * t));
    J(l2)->setDriveVelocity(PxArticulationAxis::eX, 0.2f * cosf(0.9f * t));
    J(f1)->setDriveTarget(PxArticulationAxis::eY, (s / 100) % 2 ? 0.02f : -0.02f);
    if (s == 150) {
      fj = PxFixedJointCreate(*phys, f1, PxTransform(PxVec3(0.05f, 0, 0)), bodies[2], PxTransform(PxIdentity));
      fj->setName("/World/grasp_joint");
    }
    if (s == 350 && fj) { fj->release(); fj = nullptr; }
    if (s == 420) bodies[4]->addForce(PxVec3(0, 0, 30.0f));
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // 마지막 상태 (결정성 비교용)
  FILE* f = fopen((out + "/final_state.bin").c_str(), "wb");
  for (PxRigidDynamic* b : bodies) { PxTransform p = b->getGlobalPose(); fwrite(&p, sizeof p, 1, f); }
  PxArticulationLink* links[8];
  PxU32 nl = art->getLinks(links, 8);
  for (PxU32 i = 0; i < nl; ++i) { PxTransform p = links[i]->getGlobalPose(); fwrite(&p, sizeof p, 1, f); }
  fclose(f);
  printf("완료: 서브스텝 %d, 스레드 %d, 동체 %zu, 링크 %u -> %s\n", substeps, threads, bodies.size(), nl, out.c_str());

  scene->release();
  PxCloseExtensions();
  phys->release();
  if (opvd) opvd->release();
  fnd->release();
  return 0;
}
