# joints 시험 (joints 작업자). 층 0(PhysX) = 층 1(core/joints) = 층 2(cuda/joints)
engine_test(test_joints_prep test_joints_prep.cpp jeng_api.cpp PHYSX)
engine_test(test_joints_block4 test_joints_block4.cpp jeng_api.cpp PHYSX)
# FloatV 한 칸 표현(12.7, ENG_AOS_FLOATV_ONE_LANE) 으로도 PhysX 와 비트 같은지 (엔진 번역 단위만 영향)
engine_test(test_joints_prep_1lane test_joints_prep.cpp jeng_api.cpp PHYSX)
target_compile_definitions(test_joints_prep_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)
engine_test(test_joints_block4_1lane test_joints_block4.cpp jeng_api.cpp PHYSX)
target_compile_definitions(test_joints_block4_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)
engine_test(test_tanf test_tanf.cpp)
target_compile_options(test_tanf PRIVATE -fno-builtin)
engine_test(test_joints_gpu test_joints_gpu.cu)
engine_test(test_joints_block4_gpu test_joints_block4_gpu.cu)
engine_test(test_joints_gpu_1lane test_joints_gpu.cu)
target_compile_definitions(test_joints_gpu_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)
engine_test(test_joints_block4_gpu_1lane test_joints_block4_gpu.cu)
target_compile_definitions(test_joints_block4_gpu_1lane PRIVATE ENG_AOS_FLOATV_ONE_LANE)
engine_test(test_tanf_gpu test_tanf_gpu.cu)
target_compile_options(test_tanf_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-builtin>)
engine_test(test_rigid_api test_rigid_api.cpp PHYSX)
engine_test(test_scene_query test_scene_query.cpp PHYSX)
# 공식 기록(OVD) 조인트 조사 도구 (replay/ovd.cpp 리더 재사용)
engine_test(ovd_joint_census ovd_joint_census.cpp ${CMAKE_SOURCE_DIR}/../replay/ovd.cpp PHYSX)
target_link_libraries(ovd_joint_census dl)
# 조인트 붙이기·떼기 부수효과: PhysX 내부(Sc 층) 상태를 읽는다
set(JX_INTERNAL_INCS
  ${PHYSX_ROOT}/source/common/include ${PHYSX_ROOT}/source/common/src ${PHYSX_ROOT}/source/geomutils/include ${PHYSX_ROOT}/source/geomutils/src
  ${PHYSX_ROOT}/source/geomutils/src/contact ${PHYSX_ROOT}/source/geomutils/src/pcm ${PHYSX_ROOT}/source/geomutils/src/common
  ${PHYSX_ROOT}/source/geomutils/src/mesh ${PHYSX_ROOT}/source/geomutils/src/convex ${PHYSX_ROOT}/source/geomutils/src/hf
  ${PHYSX_ROOT}/source/geomutils/src/gjk ${PHYSX_ROOT}/source/geomutils/src/intersection ${PHYSX_ROOT}/source/geomutils/src/distance
  ${PHYSX_ROOT}/source/geomutils/src/sweep ${PHYSX_ROOT}/source/geomutils/src/ccd
  ${PHYSX_ROOT}/source/lowlevel/api/include ${PHYSX_ROOT}/source/lowlevel/common/include/pipeline ${PHYSX_ROOT}/source/lowlevel/common/include/utils
  ${PHYSX_ROOT}/source/lowlevel/software/include ${PHYSX_ROOT}/source/lowlevelaabb/include ${PHYSX_ROOT}/source/lowleveldynamics/include
  ${PHYSX_ROOT}/source/lowleveldynamics/shared ${PHYSX_ROOT}/source/lowleveldynamics/src
  ${PHYSX_ROOT}/source/physx/src ${PHYSX_ROOT}/source/physxgpu/include ${PHYSX_ROOT}/source/pvd/include
  ${PHYSX_ROOT}/source/scenequery/include ${PHYSX_ROOT}/source/simulationcontroller/include ${PHYSX_ROOT}/source/simulationcontroller/src
  ${PHYSX_ROOT}/source/omnipvd ${PHYSX_ROOT}/source/physxmetadata/core/include)
engine_test(test_joint_lifecycle test_joint_lifecycle.cpp PHYSX)
target_include_directories(test_joint_lifecycle PRIVATE ${JX_INTERNAL_INCS})
target_compile_definitions(test_joint_lifecycle PRIVATE PX_PHYSX_STATIC_LIB PX_PUBLIC_RELEASE=1 PX_NVTX=0)
target_compile_options(test_joint_lifecycle PRIVATE -Wno-everything)
