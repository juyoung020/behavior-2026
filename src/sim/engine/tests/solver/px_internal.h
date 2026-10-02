// 시험 전용: PhysX 내부 자료를 읽기 위한 헤더 묶음 (최종 엔진은 PhysX 를 링크하지 않는다 — 시험 프로그램만).
// PhysX 5.6.1 checked 정적 라이브러리와 같은 정의(PX_CHECKED 등)로 컴파일해야 자료 배치가 같다.
// 표준 헤더를 먼저 넣고, 그 다음 private/protected 를 public 으로 바꿔 내부 필드를 읽는다(배치는 바뀌지 않음).
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
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "NpScene.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsIslandSim.h"
#include "PxsSimpleIslandManager.h"
#include "PxvNphaseImplementationContext.h"
#include "ScBodySim.h"
#include "ScShapeInteraction.h"
#include "ScScene.h"
#include "DyDynamicsBase.h"
#include "DyFrictionPatch.h"
#include "DyConstraint.h"
#include "DyConstraintWriteBack.h"
#include "DyContext.h"
#undef private
#undef protected
