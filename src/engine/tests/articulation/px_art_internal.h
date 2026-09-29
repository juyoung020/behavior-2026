// 시험 전용: PhysX 관절체 내부(Dy::FeatherstoneArticulation)를 직접 부르려고 내부 헤더를 묶는다. 최종 엔진은 PhysX 를 링크하지 않는다.
// PhysX checked 정적 라이브러리와 같은 정의(PX_CHECKED 등)로 컴파일해야 자료 배치가 같다. private/protected 를 public 으로 (배치 불변).
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "PxPhysicsAPI.h"

#define private public
#define protected public
#include "NpArticulationReducedCoordinate.h"
#include "ScArticulationSim.h"
#include "DyFeatherstoneArticulation.h"
#undef private
#undef protected

inline physx::Dy::FeatherstoneArticulation* llArticulation(physx::PxArticulationReducedCoordinate* a) {
  return static_cast<physx::NpArticulationReducedCoordinate*>(a)->getCore().getSim()->getLowLevelArticulation();
}
