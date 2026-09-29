// CPU 재생기에는 GPU 모듈이 없다. PhysXPvdSDK(PxPvdImpl.cpp:108,172)가 부르는 GPU 프로파일러 등록 함수를 빈 함수로 채운다.
namespace physx { class PxProfilerCallback; }
extern "C" void PxSetPhysXGpuProfilerCallback(physx::PxProfilerCallback*) {}
