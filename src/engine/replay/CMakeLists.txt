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
  target_include_directories(${t} PRIVATE ${PHYSX_ROOT}/include ${PHYSX_ROOT}/pvdruntime/include)
  target_compile_definitions(${t} PRIVATE ${PX_DEFS})
  target_compile_options(${t} PRIVATE ${COMMON_FLAGS})
  target_link_libraries(${t} ovd ${PHYSX_LIBS})
endforeach()

# S1 (문서 15절): 제어기 오프라인 비교 — OVD 만 읽는다 (PhysX 링크 없음), omni 제어기 헤더 사용
add_executable(s1_ctrl s1_ctrl.cpp)
target_link_libraries(s1_ctrl ovd)
target_include_directories(s1_ctrl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(s1_ctrl PRIVATE -O2 -ffp-contract=off -fno-fast-math)
