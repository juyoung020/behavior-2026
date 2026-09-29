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
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing> $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>)
endfunction()

contact_test(test_pcm_convex test_pcm_convex.cpp PHYSX)
contact_test(test_pcm_prims test_pcm_prims.cpp PHYSX)
contact_test(test_pcm_convex_gpu test_pcm_convex_gpu.cpp pcm_gpu_launch.cu PHYSX)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(test_pcm_convex_gpu CUDA::cudart)
target_compile_options(test_pcm_convex_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -Xcudafe=--diag_suppress=177>)
target_compile_options(test_pcm_convex_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=550>)
engine_test(test_acosf_exhaustive test_acosf_exhaustive.cpp)
target_compile_options(test_acosf_exhaustive PRIVATE -fno-builtin)
