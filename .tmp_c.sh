cd /tmp && cat > t_aos.cpp <<'X'
#include "core/contact/px/aos.h"
int main(){ using namespace eng::px; using namespace eng::px::aos; Vec3V a=V3LoadU(PxVec3(1,2,3)); FloatV d=V3Dot(a,a); PxF32 f; FStore(d,&f); return f==14.0f?0:1; }
X
clang++ -std=c++17 -O2 -ffp-contract=off -I/mnt/c/behavior-2026/src/engine t_aos.cpp -o t_aos 2>&1 | grep -E "error|warning: unused" | head -30; ./t_aos; echo "rc=$?"
cp t_aos.cpp t_aos.cu; /usr/local/cuda-12.8/bin/nvcc -std=c++17 -arch=sm_120 -fmad=false --expt-relaxed-constexpr -I/mnt/c/behavior-2026/src/engine -c t_aos.cu -o t_aos.o 2>&1 | grep -E "error" | head -30
