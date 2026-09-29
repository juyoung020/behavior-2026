# common 시험 (리드)
engine_test(test_sincosf_host test_sincosf_host.cpp)
target_compile_options(test_sincosf_host PRIVATE -fno-builtin)
# aos 흉내 = 진짜 PhysX aos (함수마다 난수 입력 비트 비교). 생성: scripts/gen_aos_test.py
engine_test(test_aos_diff aos_diff_main.cpp aos_diff_eng.cpp aos_diff_px.cpp PHYSX)
