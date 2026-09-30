# 층 0 도구: OmniPVD(.ovd) 읽기·요약·재생. PhysX 5.6.1 (checked 빌드, ~/engine-deps) 에 링크한다.
# 이 도구들은 "정답지"를 만드는 비교 기준일 뿐, 최종 엔진은 PhysX 를 링크하지 않는다 (docs/엔진_자체구현.md).
#   cmake -S /mnt/c/behavior-2026/src/engine/replay -B ~/engine-build/replay -DCMAKE_BUILD_TYPE=Release && cmake --build ~/engine-build/replay -j
cmake_minimum_required(VERSION 3.16)
project(engine_replay CXX)

set(CMAKE_CXX_STANDARD 17)
set(PHYSX_ROOT "$ENV{HOME}/engine-deps/physx-107.3-omni/physx" CACHE PATH "PhysX SDK 소스 루트")
set(PHYSX_CONFIG "checked" CACHE STRING "PhysX 빌드 설정 (공식 omni.physx 는 checked: PX_CHECK 문자열이 DLL 에 있음)")
set(PHYSX_LIB "${PHYSX_ROOT}/bin/linux.x86_64/${PHYSX_CONFIG}")

if(PHYSX_CONFIG STREQUAL "checked")
  set(PX_DEFS NDEBUG PX_CHECKED=1 PX_SUPPORT_PVD=1 PX_SUPPORT_OMNI_PVD=1)
elseif(PHYSX_CONFIG STREQUAL "release")
  set(PX_DEFS NDEBUG PX_SUPPORT_PVD=0 PX_SUPPORT_OMNI_PVD=0)
else()
  set(PX_DEFS NDEBUG PX_PROFILE=1 PX_SUPPORT_PVD=1 PX_SUPPORT_OMNI_PVD=1)
endif()

# 우리 코드에서는 물리 값에 산술을 하지 않는다(복사만). 그래도 부동소수 축약·재배열을 막아 둔다.
set(COMMON_FLAGS -O2 -ffp-contract=off -fno-fast-math)

add_library(ovd STATIC ovd.cpp)
target_include_directories(ovd PUBLIC ${PHYSX_ROOT}/pvdruntime/include)
target_compile_options(ovd PRIVATE ${COMMON_FLAGS})
target_link_libraries(ovd PUBLIC dl)

foreach(t ovd_dump ovd_diff ovd_pairs)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} ovd)
  target_compile_options(${t} PRIVATE ${COMMON_FLAGS})
endforeach()

set(PHYSX_LIBS
  ${PHYSX_LIB}/libPhysXExtensions_static_64.a
  ${PHYSX_LIB}/libPhysX_static_64.a
  ${PHYSX_LIB}/libPhysXPvdSDK_static_64.a
  ${PHYSX_LIB}/libPhysXCooking_static_64.a
  ${PHYSX_LIB}/libPhysXCommon_static_64.a
  ${PHYSX_LIB}/libPhysXFoundation_static_64.a
  pthread dl)

foreach(t ovd_selftest ovd_replay)
  add_executable(${t} ${t}.cpp px_gpu_stub.cpp)
  target_include_directories(${t} PRIVATE ${PHYSX_ROOT}/include ${PHYSX_ROOT}/pvdruntime/include ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_compile_definitions(${t} PRIVATE ${PX_DEFS})
  target_compile_options(${t} PRIVATE ${COMMON_FLAGS})
  target_link_libraries(${t} ovd ${PHYSX_LIBS})
endforeach()

# S1 (문서 15절): 제어기 오프라인 비교 — OVD 만 읽는다 (PhysX 링크 없음), omni 제어기 헤더 사용
add_executable(s1_ctrl s1_ctrl.cpp)
target_link_libraries(s1_ctrl ovd)
target_include_directories(s1_ctrl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(s1_ctrl PRIVATE -O2 -ffp-contract=off -fno-fast-math)

# S4 v0 (문서 15절): 엔진 파이썬 바인딩의 C 입구 (ctypes). PhysX 정적 라이브러리가 -fPIC 가 아니면 링크가 실패한다 -> 그때는 PhysX 를 PIC 로 다시 빌드
add_library(engine_capi SHARED engine_capi.cpp px_gpu_stub.cpp)
target_include_directories(engine_capi PRIVATE ${PHYSX_ROOT}/include ${PHYSX_ROOT}/pvdruntime/include ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(engine_capi PRIVATE ${PX_DEFS})
target_compile_options(engine_capi PRIVATE ${COMMON_FLAGS})
set_target_properties(ovd PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(engine_capi ovd ${PHYSX_LIBS})

# G1a 그림자 시험 (문서 15절): ovd_replay + g1_shadow.cpp (PhysX 내부 헤더 + 우리 모듈), 실행 파일 ovd_replay_g1
set(G1_PX_INC
  ${PHYSX_ROOT}/source/physx/src ${PHYSX_ROOT}/source/physx/src/buffering ${PHYSX_ROOT}/source/physx/src/device
  ${PHYSX_ROOT}/source/simulationcontroller/include ${PHYSX_ROOT}/source/simulationcontroller/src
  ${PHYSX_ROOT}/source/lowlevel/common/include/collision ${PHYSX_ROOT}/source/lowlevel/common/include/pipeline
  ${PHYSX_ROOT}/source/lowlevel/common/include/utils ${PHYSX_ROOT}/source/lowlevel/api/include ${PHYSX_ROOT}/source/lowlevel/software/include
  ${PHYSX_ROOT}/source/lowleveldynamics/include ${PHYSX_ROOT}/source/lowlevelaabb/include ${PHYSX_ROOT}/source/scenequery/include
  ${PHYSX_ROOT}/source/physxmetadata/core/include ${PHYSX_ROOT}/source/pvd/include ${PHYSX_ROOT}/source/physxgpu/include
  ${PHYSX_ROOT}/source/omnipvd ${PHYSX_ROOT}/source/physxextensions/src ${PHYSX_ROOT}/source/filebuf/include
  ${PHYSX_ROOT}/source/geomutils/include ${PHYSX_ROOT}/source/geomutils/src ${PHYSX_ROOT}/source/geomutils/src/pcm
  ${PHYSX_ROOT}/source/geomutils/src/gjk ${PHYSX_ROOT}/source/geomutils/src/convex ${PHYSX_ROOT}/source/geomutils/src/contact
  ${PHYSX_ROOT}/source/geomutils/src/common ${PHYSX_ROOT}/source/geomutils/src/mesh ${PHYSX_ROOT}/source/geomutils/src/intersection
  ${PHYSX_ROOT}/source/geomutils/src/distance ${PHYSX_ROOT}/source/geomutils/src/hf
  ${PHYSX_ROOT}/source/common/include ${PHYSX_ROOT}/source/common/src ${PHYSX_ROOT}/source/foundation/include)
# G1_CORE_ROOT: solver 를 가져올 엔진 뿌리 (기본 = 이 저장소 src/engine). 다른 작업자가 core/solver 를 고치는 중이면 커밋본을 풀어 둔 곳을 준다.
set(G1_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "G1 solver 그림자가 쓸 src/engine 뿌리")
add_executable(ovd_replay_g1 ovd_replay.cpp g1_shadow.cpp g1_solver.cpp g1_art.cpp g1_dump.cpp g1_islands.cpp g1_bp.cpp g1_pairs.cpp g1_sc.cpp g1_sc_math.cpp g1_scene.cpp g1_loop.cpp g1_host.cpp ${G1_CORE_ROOT}/core/solver/solver_host.cpp px_gpu_stub.cpp)
target_include_directories(ovd_replay_g1 BEFORE PRIVATE ${G1_CORE_ROOT})  # "core/..." 꼴 include 도 같은 뿌리에서
target_include_directories(ovd_replay_g1 PRIVATE ${PHYSX_ROOT}/include ${PHYSX_ROOT}/pvdruntime/include ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_source_files_properties(g1_shadow.cpp PROPERTIES INCLUDE_DIRECTORIES "${G1_PX_INC}"
  COMPILE_DEFINITIONS "PX_PUBLIC_RELEASE=1;PX_NVTX=0" COMPILE_OPTIONS "-fno-strict-aliasing;-Wno-macro-redefined")
# G1 solver 그림자: PhysX 섬·Dy 내부를 읽는다 (tests/solver 의 PX_INTERNAL_INCS 와 같은 경로)
set(G1S_PX_INC ${G1_PX_INC}
  ${PHYSX_ROOT}/source/geomutils/src/sweep ${PHYSX_ROOT}/source/geomutils/src/ccd ${PHYSX_ROOT}/source/lowlevel/api/include
  ${PHYSX_ROOT}/source/lowleveldynamics/shared ${PHYSX_ROOT}/source/lowleveldynamics/src ${PHYSX_ROOT}/source/immediatemode/include)
set_source_files_properties(g1_solver.cpp PROPERTIES INCLUDE_DIRECTORIES "${G1S_PX_INC}"
  COMPILE_DEFINITIONS "PX_PHYSX_STATIC_LIB;PX_PUBLIC_RELEASE=1;PX_NVTX=0;G1_SOLVER_IO=\"${G1_CORE_ROOT}/core/solver/solver_io.h\"" COMPILE_OPTIONS "-Wno-macro-redefined")
set_source_files_properties(g1_art.cpp g1_dump.cpp g1_islands.cpp g1_bp.cpp g1_pairs.cpp g1_sc.cpp g1_scene.cpp PROPERTIES INCLUDE_DIRECTORIES "${G1S_PX_INC};${PHYSX_ROOT}/source/lowlevelaabb/src"
  COMPILE_DEFINITIONS "PX_PHYSX_STATIC_LIB;PX_PUBLIC_RELEASE=1;PX_NVTX=0" COMPILE_OPTIONS "-Wno-everything")
target_compile_definitions(ovd_replay_g1 PRIVATE ${PX_DEFS})
target_compile_options(ovd_replay_g1 PRIVATE ${COMMON_FLAGS})
target_link_libraries(ovd_replay_g1 ovd ${PHYSX_LIBS})
# 진단: PhysX 접촉 준비(4 개 묶음/하나씩) 호출을 가로챈다 (g1_solver.cpp) + 섬 관리자 호출 기록 (g1_islands.cpp, tests/solver test_islands 와 같은 목록)
target_link_options(ovd_replay_g1 PRIVATE
  -Wl,--wrap=_ZN5physx2Bp13BroadPhaseABPC1Ejjjmb
  -Wl,--wrap=_ZN5physx2Sc8BodyCore17setCMassLocalPoseERKNS_12PxTransformTIfEE
  -Wl,--wrap=_ZN5physx2Sc8BodyCore13setBody2WorldERKNS_12PxTransformTIfEE
  -Wl,--wrap=_ZN5physx2Sc5Scene7addBodyERNS0_8BodyCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3Eb
  -Wl,--wrap=_ZN5physx2Sc5Scene9addStaticERNS0_10StaticCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3E
  -Wl,--wrap=_ZN5physx2Sc5Scene10removeBodyERNS0_8BodyCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb
  -Wl,--wrap=_ZN5physx2Sc5Scene12removeStaticERNS0_10StaticCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb
  -Wl,--wrap=_ZN5physx2Sc12ShapeSimBase18reinsertBroadPhaseEv
  -Wl,--wrap=_ZN5physx2Sc9RigidCore15addShapeToSceneERNS0_9ShapeCoreE
  -Wl,--wrap=_ZN5physx2Sc9RigidCore20removeShapeFromSceneERNS0_9ShapeCoreEb
  -Wl,--wrap=_ZN5physx2Sc9RigidCore13onShapeChangeERNS0_9ShapeCoreENS_7PxFlagsINS0_21ShapeChangeNotifyFlag4EnumEjEE
  -Wl,--wrap=_ZN5physx2Sc12ShapeSimBase12onFlagChangeENS_7PxFlagsINS_11PxShapeFlag4EnumEhEE
  -Wl,--wrap=_ZNK5physx2Sc10NPhaseCore17runOverlapFiltersEjPNS_2Bp11AABBOverlapEPNS0_10FilterInfoERjS7_
  -Wl,--wrap=_ZN5physx2Sc10NPhaseCore23onTriggerOverlapCreatedEPKNS_2Bp11AABBOverlapEj
  -Wl,--wrap=_ZN5physx2Sc10NPhaseCore16onOverlapRemovedEPNS0_10ElementSimES3_jPvRNS_31PxsContactManagerOutputIteratorE
  -Wl,--wrap=_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE
  -Wl,--wrap=_ZN5physx2Sc16ShapeInteraction12onDeactivateEv
  -Wl,--wrap=_ZNK5physx2Sc16ShapeInteraction23resetManagerCachedStateEv
  -Wl,--wrap=_ZN5physx2Sc16ShapeInteraction26onShapeChangeWhileSleepingEb
  -Wl,--wrap=_ZN5physx10PxsContext22fillManagerTouchEventsEPNS_27PxvContactManagerTouchEventERjS2_S3_S2_S3_
  -Wl,--wrap=_ZN5physx2Bp11AABBManagerC1ERNS0_10BroadPhaseERNS0_11BoundsArrayERNS_7PxArrayIfNS_17PxPinnedAllocatorIfEEEEjjRNS_18PxVirtualAllocatorEmNS_19PxPairFilteringMode4EnumESE_
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager7addNodeEbbNS0_4Node8NodeTypeEPv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager13addConstraintEPNS_2Dy10ConstraintENS_11PxNodeIndexES5_PNS_2Sc11InteractionE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager12activateNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14putNodeToSleepENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager12setKinematicENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager10setDynamicENS_11PxNodeIndexE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16removeConnectionEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager19setEdgeDisconnectedEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14deactivateEdgeEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16clearEdgeRigidCMEj
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager18firstPassIslandGenEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager31additionalSpeculativeActivationEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart1Ev
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart2Ev
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager19secondPassIslandGenEv
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager18thirdPassIslandGenEPNS_10PxBaseTaskE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE
  -Wl,--wrap=_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE
  -Wl,--wrap=_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj
  -Wl,--wrap=_ZN5physx2IG9IslandSim20removeDestroyedEdgesEv
  -Wl,--wrap=_ZN5physx2IG9IslandSim16processLostEdgesERKNS_7PxArrayINS_11PxNodeIndexENS_21PxReflectionAllocatorIS3_EEEEbbj
  -Wl,--wrap=_ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE
  -Wl,--wrap=_ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE)
