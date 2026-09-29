# articulation 시험 (articulation 작업자). 층 1: 무작위 관절체 PhysX = 손으로 짠 C++ (비트), 층 2: CUDA 판 N 개 = 층 1 = PhysX
engine_test(test_articulation test_articulation.cpp PHYSX)
engine_test(test_articulation_gpu test_articulation_gpu.cu PHYSX)
# R1Pro 모양(링크 42, dof 28) 용량판
engine_test(test_articulation_gpu_r1 test_articulation_gpu.cu PHYSX)
target_compile_definitions(test_articulation_gpu_r1 PRIVATE ENG_ART_MAX_LINKS=48 ENG_ART_MAX_DOFS=40 ENG_ART_MAX_PATH=640)

# 결합점 함수 시험: PhysX 내부(Dy::FeatherstoneArticulation)를 직접 부른다 -> 내부 헤더 경로·정의
set(ART_PX_INTERNAL_INCS
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
engine_test(test_articulation_coupling test_articulation_coupling.cpp PHYSX)
target_include_directories(test_articulation_coupling PRIVATE ${ART_PX_INTERNAL_INCS})
target_compile_definitions(test_articulation_coupling PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_articulation_coupling PRIVATE -Wno-everything)

# 처리량 실험: 레지스터 상한 (기본은 스레드당 210 개 -> SM 당 워프 약 8 개). 결과 비트에는 영향 없음(할당만 다름)
foreach(r 128 96 64)
  engine_test(test_articulation_gpu_r1_m${r} test_articulation_gpu.cu PHYSX)
  target_compile_definitions(test_articulation_gpu_r1_m${r} PRIVATE ENG_ART_MAX_LINKS=48 ENG_ART_MAX_DOFS=40 ENG_ART_MAX_PATH=640)
  target_compile_options(test_articulation_gpu_r1_m${r} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=${r}>)
endforeach()
