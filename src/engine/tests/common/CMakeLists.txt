# common 시험 (리드)
engine_test(test_sincosf_host test_sincosf_host.cpp)
target_compile_options(test_sincosf_host PRIVATE -fno-builtin)
