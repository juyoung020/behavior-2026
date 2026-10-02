# libsgrt.so — ovdet(YOLOE TensorRT) + scenemap(slam2d·objmap·Spark-DSG 저장) 을 한 C ABI 로 (include/sgrt.h).
#   cmake -S src/scene_graph/runtime -B ~/sgrt_build && cmake --build ~/sgrt_build -j
cmake_minimum_required(VERSION 3.18)
project(sgrt LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scenemap ${CMAKE_BINARY_DIR}/scenemap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ovdet ${CMAKE_BINARY_DIR}/ovdet)

add_library(sgrt SHARED src/sgrt.cpp)
target_include_directories(sgrt PUBLIC include)
target_link_libraries(sgrt PRIVATE scenemap ovdet)
target_compile_options(sgrt PRIVATE -Wall -Wextra)
