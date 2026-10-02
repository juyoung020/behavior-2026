// PhysX AABB 관리자(Bp::AABBManager) 의 우리 판 (eng::px::Bp::AABBManager). PhysX 링크 없음. 층 1(호스트) 판.
// 원본 lowlevelaabb/src/BpAABBManager.cpp·BpAABBManagerBase.cpp 와 foundation 해시 컨테이너(PxHashInternals.h 등)를
// gen/translate.py 로 기계 번역(gen_aabb.inc). 묶음(aggregate) 쌍 표의 해시 순회 순서가 새 겹침 순서를 정하므로
// 해시 컨테이너도 원본 그대로 옮겼다. 작업(task)은 bp_support.inc 처럼 한 스레드로 차례대로 돈다.
#pragma once
#include <cstdio>
#include <vector>

#include "core/contact/px/bp.h"

namespace eng {
#include "core/contact/px/aabb_support.inc"
#include "core/contact/px/gen_aabb.inc"
}  // namespace eng
