# libsgrt.so — ovdet(YOLOE TensorRT) + scenemap(slam2d·objmap·Spark-DSG 저장) 을 한 C ABI 로 (include/sgrt.h).
#   cmake -S src/scene_graph/runtime -B ~/sgrt_build && cmake --build ~/sgrt_build -j
cmake_minimum_required(VERSION 3.18)
# CUDA: ovdet 과 같은 툴킷(12.8, sm_120). best view 자르기 커널(src/crop.cu)
set(OVDET_CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CMAKE_CUDA_COMPILER ${OVDET_CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "GPU architectures")
project(sgrt LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scenemap ${CMAKE_BINARY_DIR}/scenemap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ovdet ${CMAKE_BINARY_DIR}/ovdet)

add_library(sgrt SHARED src/sgrt.cpp src/crop.cu)
target_include_directories(sgrt PUBLIC include)
set_target_properties(sgrt PROPERTIES CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(sgrt PRIVATE scenemap ovdet)
target_compile_options(sgrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
