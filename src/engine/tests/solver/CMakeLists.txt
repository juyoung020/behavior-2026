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

engine_test(test_contact_solver test_contact_solver.cpp PHYSX)
target_include_directories(test_contact_solver PRIVATE ${PX_INTERNAL_INCS})
target_compile_definitions(test_contact_solver PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_contact_solver PRIVATE -Wno-everything)
