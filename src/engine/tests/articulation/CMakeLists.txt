# articulation 시험 (articulation 작업자). 층 1: 무작위 관절체 PhysX = 손으로 짠 C++ (비트), 층 2: CUDA 판 N 개 = 층 1 = PhysX
engine_test(test_articulation test_articulation.cpp PHYSX)
engine_test(test_articulation_gpu test_articulation_gpu.cu PHYSX)
