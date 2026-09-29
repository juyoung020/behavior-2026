# omni 시험 (omni 작업자). 정답은 PhysX 가 아니라 공식 OmniGibson/bddl3 파이썬 (gen_*.py 가 ~/engine-data/omni/ 에 적음)
engine_test(test_bddl test_bddl.cpp)
engine_test(test_controllers test_controllers.cpp)
engine_test(test_states test_states.cpp)

# 층 2 (CUDA). omni 는 FTZ 를 끈다: 제어기(numpy)·물체 상태(warp) 모두 비정규수를 그대로 두기 때문 (PhysX 층의 -ftz=true 와 다름)
add_executable(test_omni_gpu test_omni_gpu.cu)
target_include_directories(test_omni_gpu PRIVATE ${CMAKE_SOURCE_DIR}/..)
target_compile_options(test_omni_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false --expt-relaxed-constexpr -Xcompiler=-ffp-contract=off>)
engine_test(test_ag test_ag.cpp)
engine_test(test_agframe test_agframe.cpp)
