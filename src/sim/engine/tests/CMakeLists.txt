# 엔진 시험 (층 1 C++ / 층 2 CUDA vs 층 0 PhysX 정답지). 정답지 비교 때문에 "시험 프로그램만" PhysX 에 링크한다.
#   export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
#   cmake -S /mnt/c/behavior-2026/src/sim/engine/tests -B ~/engine-build/tests -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++
#   cmake --build ~/engine-build/tests -j
# 모듈마다 tests/<모듈>/CMakeLists.txt 를 두고 engine_test() 로 시험을 더한다 (이 파일은 리드가 관리, 모듈은 자기 폴더만).
cmake_minimum_required(VERSION 3.18)
# project() 앞에서 정한다: project() 가 툴킷 기본값(CUDA 12 는 52, CUDA 13 은 75)을 넣으면 사용자 지정과 구별이 안 된다.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)  # RTX 5070 Ti. 다른 장비는 -DCMAKE_CUDA_ARCHITECTURES=86 등
endif()
project(engine_tests CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(PHYSX_ROOT "$ENV{HOME}/engine-deps/physx-107.3-omni/physx" CACHE PATH "PhysX 5.6.1 소스·빌드 (정답지)")
set(PHYSX_CONFIG "checked" CACHE STRING "공식 omni.physx 와 같은 checked")
set(PHYSX_LIB "${PHYSX_ROOT}/bin/linux.x86_64/${PHYSX_CONFIG}")
set(PX_DEFS NDEBUG PX_CHECKED=1 PX_SUPPORT_PVD=1 PX_SUPPORT_OMNI_PVD=1)
set(PX_LIBS
  ${PHYSX_LIB}/libPhysXExtensions_static_64.a ${PHYSX_LIB}/libPhysX_static_64.a ${PHYSX_LIB}/libPhysXPvdSDK_static_64.a
  ${PHYSX_LIB}/libPhysXCooking_static_64.a ${PHYSX_LIB}/libPhysXCommon_static_64.a ${PHYSX_LIB}/libPhysXFoundation_static_64.a
  pthread dl)

# 비트 규칙 (docs/엔진_자체구현.md 12.2): 축약·재배열 금지. CUDA 는 IEEE 나눗셈·제곱근, FTZ(PhysX PxSIMDGuard 와 같게)
set(ENGINE_CXX_FLAGS -O2 -ffp-contract=off -fno-fast-math)
set(ENGINE_CUDA_FLAGS -O2 -fmad=false -prec-div=true -prec-sqrt=true -ftz=true --expt-relaxed-constexpr -Xcompiler=-ffp-contract=off)

# engine_test(<이름> <소스...> [PHYSX])  — PHYSX 를 주면 정답지 PhysX 에 링크
function(engine_test name)
  cmake_parse_arguments(ET "PHYSX" "" "" ${ARGN})
  set(srcs ${ET_UNPARSED_ARGUMENTS})
  if(ET_PHYSX)
    list(APPEND srcs ${CMAKE_SOURCE_DIR}/../replay/px_gpu_stub.cpp)
  endif()
  add_executable(${name} ${srcs})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/..)
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${ENGINE_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${ENGINE_CXX_FLAGS}>)
  target_link_libraries(${name} pthread m)
  if(ET_PHYSX)
    target_include_directories(${name} PRIVATE ${PHYSX_ROOT}/include ${PHYSX_ROOT}/pvdruntime/include)
    target_compile_definitions(${name} PRIVATE ${PX_DEFS})
    target_link_libraries(${name} ${PX_LIBS})
  endif()
endfunction()

foreach(m common contact solver articulation joints omni render particles integration scene)
  if(EXISTS ${CMAKE_SOURCE_DIR}/${m}/CMakeLists.txt)
    add_subdirectory(${m})
  endif()
endforeach()
