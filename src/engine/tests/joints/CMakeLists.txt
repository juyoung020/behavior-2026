# joints 시험 (joints 작업자). 층 0(PhysX) = 층 1(core/joints) = 층 2(cuda/joints)
engine_test(test_joints_prep test_joints_prep.cpp jeng_api.cpp PHYSX)
