# contact 시험 (contact 작업자). PhysX 내부 함수(Gu::pcm*)를 직접 불러 함수 단위로 비교하므로 내부 헤더 경로를 더한다.
set(PX_INTERNAL_INC
  ${PHYSX_ROOT}/source/geomutils/include ${PHYSX_ROOT}/source/geomutils/src ${PHYSX_ROOT}/source/geomutils/src/pcm
  ${PHYSX_ROOT}/source/geomutils/src/gjk ${PHYSX_ROOT}/source/geomutils/src/convex ${PHYSX_ROOT}/source/geomutils/src/contact
  ${PHYSX_ROOT}/source/geomutils/src/common ${PHYSX_ROOT}/source/geomutils/src/mesh ${PHYSX_ROOT}/source/geomutils/src/intersection
  ${PHYSX_ROOT}/source/geomutils/src/distance ${PHYSX_ROOT}/source/geomutils/src/hf
  ${PHYSX_ROOT}/source/common/include ${PHYSX_ROOT}/source/common/src ${PHYSX_ROOT}/source/foundation/include
  ${PHYSX_ROOT}/source/lowlevel/api/include ${PHYSX_ROOT}/source/lowlevel/common/include/pipeline
  ${PHYSX_ROOT}/source/lowlevel/common/include/utils ${PHYSX_ROOT}/source/lowlevel/software/include)

function(contact_test name)
  engine_test(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${PX_INTERNAL_INC})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing -Wno-macro-redefined>)
endfunction()

contact_test(test_pcm_convex test_pcm_convex.cpp PHYSX)
