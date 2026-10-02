# solver 시험 (solver 작업자). 씨앗: 자유 강체 층0=층1=층2
engine_test(test_free_bodies test_free_bodies.cpp PHYSX)
engine_test(test_free_bodies_gpu test_free_bodies_gpu.cu PHYSX)

# PhysX 내부 자료(섬·접촉 관리자 출력)를 읽는 시험: 내부 헤더 경로와 정적 라이브러리 정의가 더 필요하다
set(PX_INTERNAL_INCS
  ${PHYSX_ROOT}/source/common/include ${PHYSX_ROOT}/source/common/src
  ${PHYSX_ROOT}/source/geomutils/include ${PHYSX_ROOT}/source/geomutils/src ${PHYSX_ROOT}/source/geomutils/src/contact
  ${PHYSX_ROOT}/source/geomutils/src/pcm ${PHYSX_ROOT}/source/geomutils/src/common ${PHYSX_ROOT}/source/geomutils/src/mesh
  ${PHYSX_ROOT}/source/geomutils/src/convex ${PHYSX_ROOT}/source/geomutils/src/hf ${PHYSX_ROOT}/source/geomutils/src/gjk
  ${PHYSX_ROOT}/source/geomutils/src/intersection ${PHYSX_ROOT}/source/geomutils/src/distance ${PHYSX_ROOT}/source/geomutils/src/sweep
  ${PHYSX_ROOT}/source/geomutils/src/ccd
  ${PHYSX_ROOT}/source/lowlevel/api/include ${PHYSX_ROOT}/source/lowlevel/common/include/pipeline
  ${PHYSX_ROOT}/source/lowlevel/common/include/utils ${PHYSX_ROOT}/source/lowlevel/software/include
  ${PHYSX_ROOT}/source/lowlevelaabb/include ${PHYSX_ROOT}/source/lowleveldynamics/include ${PHYSX_ROOT}/source/lowleveldynamics/shared
  ${PHYSX_ROOT}/source/lowleveldynamics/src
  ${PHYSX_ROOT}/source/physx/src ${PHYSX_ROOT}/source/physxgpu/include ${PHYSX_ROOT}/source/pvd/include
  ${PHYSX_ROOT}/source/scenequery/include ${PHYSX_ROOT}/source/simulationcontroller/include ${PHYSX_ROOT}/source/simulationcontroller/src
  ${PHYSX_ROOT}/source/omnipvd ${PHYSX_ROOT}/source/immediatemode/include ${PHYSX_ROOT}/source/physxmetadata/core/include)

# 접촉 관리자가 새로 걸리는 호출만 가로챈다(새 관리자 = 마찰 패치 0, 시험 비계용)
set(CS_WRAPS
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE
  -Wl,--wrap=_ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE
  -Wl,--wrap=_ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE)
engine_test(test_contact_solver test_contact_solver.cpp ${CMAKE_SOURCE_DIR}/../core/solver/solver_host.cpp PHYSX)
target_include_directories(test_contact_solver PRIVATE ${PX_INTERNAL_INCS})
target_compile_definitions(test_contact_solver PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_contact_solver PRIVATE -Wno-everything)
target_link_options(test_contact_solver PRIVATE ${CS_WRAPS})

# 층 2: 입력 흐름 파일(test_contact_solver --dump)로 판 N 개를 GPU 에서 (PhysX 링크 없음)
engine_test(test_contact_solver_gpu test_contact_solver_gpu.cu)

# 섬 관리: Sc 층이 섬 관리자에 하는 호출을 링커 --wrap 으로 기록해 손으로 짠 섬 관리에 재생하고 내부 상태를 비교
engine_test(test_islands test_islands.cpp PHYSX)
target_include_directories(test_islands PRIVATE ${PX_INTERNAL_INCS})
target_compile_definitions(test_islands PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_islands PRIVATE -Wno-everything)
target_link_options(test_islands PRIVATE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager10setDynamicENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager12activateNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager12setKinematicENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager13addConstraintEPNS_2Dy10ConstraintENS_11PxNodeIndexES5_PNS_2Sc11InteractionE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14deactivateEdgeEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14putNodeToSleepENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16clearEdgeRigidCMEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16removeConnectionEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager18firstPassIslandGenEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager18thirdPassIslandGenEPNS_10PxBaseTaskE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager19secondPassIslandGenEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager19setEdgeDisconnectedEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart1Ev
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart2Ev
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager31additionalSpeculativeActivationEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager7addNodeEbbNS0_4Node8NodeTypeEPv
  -Wl,--wrap=_ZN5physx2IG9IslandSim16processLostEdgesERKNS_7PxArrayINS_11PxNodeIndexENS_21PxReflectionAllocatorIS3_EEEEbbj
  -Wl,--wrap=_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj
  -Wl,--wrap=_ZN5physx2IG9IslandSim20removeDestroyedEdgesEv
)

# 같은 층 1 시험을 FloatV 한 칸 표현(-DENG_AOS_FLOATV_ONE_LANE, docs 12.7)으로 (aos 는 solver_host.cpp 번역 단위에만 있다)
engine_test(test_contact_solver_1lane test_contact_solver.cpp ${CMAKE_SOURCE_DIR}/../core/solver/solver_host.cpp PHYSX)
target_include_directories(test_contact_solver_1lane PRIVATE ${PX_INTERNAL_INCS})
target_compile_definitions(test_contact_solver_1lane PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_contact_solver_1lane PRIVATE -Wno-everything)
target_compile_definitions(test_contact_solver_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)
target_link_options(test_contact_solver_1lane PRIVATE ${CS_WRAPS})
engine_test(test_contact_solver_gpu_1lane test_contact_solver_gpu.cu)
target_compile_definitions(test_contact_solver_gpu_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)

# 관절체 결합 (09-30, engine-solver-art): 바닥 위 무작위 관절체·R1Pro·강체·링크 조인트 섬을 PhysX 스냅샷으로 풀어 비교
engine_test(test_art_solver test_art_solver.cpp ${CMAKE_SOURCE_DIR}/../core/solver/solver_host.cpp PHYSX)
target_include_directories(test_art_solver PRIVATE ${PX_INTERNAL_INCS})
target_compile_definitions(test_art_solver PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_art_solver PRIVATE -Wno-everything)
list(SUBLIST CS_WRAPS 0 3 ART_WRAPS)  # 새 관리자 표시 가로채기만 (준비 진단 가로채기는 안 씀)
target_link_options(test_art_solver PRIVATE ${ART_WRAPS})

# 층 2: test_art_solver --dump 흐름으로 판 N 개를 GPU 에서 (PhysX 링크 없음)
engine_test(test_art_solver_gpu test_art_solver_gpu.cu ${CMAKE_SOURCE_DIR}/../core/solver/solver_gpu.cu)  # --api: 공용 진입 gpuSolveBatch
target_compile_options(test_art_solver_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011,20014 -lineinfo>)  # 장치에서 호스트 함수 부름 = 오류 (19.6 교훈)

# contact 장면 한 서브스텝(core/contact/scene_step.h) 빌드 확인
engine_test(test_contact_step_build test_contact_step_build.cpp)
