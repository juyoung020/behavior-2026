# 층 1 시험: 손으로 짠 엔진(src/engine/core, PhysX 를 링크하지 않는 헤더) vs PhysX 정답지.
# 정답지 비교를 위해 "시험 프로그램만" PhysX 에 링크한다.
#   cmake -S /mnt/c/behavior-2026/src/engine/tests -B ~/engine-build/tests && cmake --build ~/engine-build/tests -j
cmake_minimum_required(VERSION 3.16)
project(engine_tests CXX)
set(CMAKE_CXX_STANDARD 17)
set(PHYSX_ROOT "$ENV{HOME}/engine-deps/physx-107.3-omni/physx" CACHE PATH "")
set(PHYSX_CONFIG "checked" CACHE STRING "")
set(PHYSX_LIB "${PHYSX_ROOT}/bin/linux.x86_64/${PHYSX_CONFIG}")
set(PX_DEFS NDEBUG PX_CHECKED=1 PX_SUPPORT_PVD=1 PX_SUPPORT_OMNI_PVD=1)
# 엔진 코드는 부동소수 축약·재배열 금지 (비트 동일의 전제)
set(ENGINE_FLAGS -O2 -ffp-contract=off -fno-fast-math)

foreach(t test_free_bodies)
  add_executable(${t} ${t}.cpp ../replay/px_gpu_stub.cpp)
  target_include_directories(${t} PRIVATE .. ${PHYSX_ROOT}/include)
  target_compile_definitions(${t} PRIVATE ${PX_DEFS})
  target_compile_options(${t} PRIVATE ${ENGINE_FLAGS})
  target_link_libraries(${t}
    ${PHYSX_LIB}/libPhysXExtensions_static_64.a ${PHYSX_LIB}/libPhysX_static_64.a ${PHYSX_LIB}/libPhysXPvdSDK_static_64.a
    ${PHYSX_LIB}/libPhysXCommon_static_64.a ${PHYSX_LIB}/libPhysXFoundation_static_64.a pthread dl)
endforeach()
