# scene 시험 (리드): N2 네이티브 루프 — 판 N 개 물리 + 렌더 (+ 정책, G4). PhysX 링크 없음.
# 물리 부분은 .cpp(clang, env_run 과 같은 옵션 + NDEBUG), 렌더는 .cu(nvcc).
engine_test(native_loop native_loop.cpp native_render.cu ${CMAKE_SOURCE_DIR}/../core/solver/solver_host.cpp)
target_compile_definitions(native_loop PRIVATE NDEBUG)
# env_run 의 GPU 판: --gpu 로 풀이 본체를 sv::gpuSolveBatch (engine-solver-art) 로
engine_test(env_run_gpu env_run.cpp ${CMAKE_SOURCE_DIR}/../core/solver/solver_gpu.cu ${CMAKE_SOURCE_DIR}/../core/solver/solver_host.cpp)
target_compile_definitions(env_run_gpu PRIVATE NDEBUG ENV_RUN_GPU)
# 정책 (G4c): pi05 네이티브 엔진 정적 라이브러리가 있으면 native_loop 에 붙인다 (src/pi05_native/build_linux.sh 가 만듦)
set(PI05_LIB "$ENV{HOME}/pi05_native_build/libpi05.a" CACHE FILEPATH "pi05 네이티브 엔진 (libpi05.a)")
if(EXISTS ${PI05_LIB})
  target_compile_definitions(native_loop PRIVATE NL_PI05)
  target_include_directories(native_loop PRIVATE ${CMAKE_SOURCE_DIR}/../../pi05_native/include)
  target_link_libraries(native_loop ${PI05_LIB} cudart dl rt)
endif()
