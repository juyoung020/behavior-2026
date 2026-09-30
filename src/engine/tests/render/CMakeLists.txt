# render 시험 (render 작업자). PhysX 안 씀: 층 0 은 공식 RTX 영상(통계 비교), 층 1(C++) = 층 2(CUDA) 는 비트 비교.
engine_test(test_render_synth test_render_synth.cu)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_render_scene.cu)
  engine_test(test_render_scene test_render_scene.cu)
endif()
engine_test(test_render_math test_render_math.cu)
