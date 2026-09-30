cmake_minimum_required(VERSION 3.18)
project(scenemap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_library(scenemap STATIC src/scan.cpp src/grid.cpp src/slam2d.cpp src/fk.cpp)
target_include_directories(scenemap PUBLIC include)
set_target_properties(scenemap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(scenemap PRIVATE -Wall -Wextra)

add_executable(slam2d_eval tools/slam2d_eval.cpp)
target_link_libraries(slam2d_eval scenemap)

enable_testing()
add_executable(test_fk tests/test_fk.cpp)
target_link_libraries(test_fk scenemap)
add_test(NAME fk COMMAND test_fk)
