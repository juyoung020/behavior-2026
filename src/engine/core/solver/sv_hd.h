// solver 호스트·GPU 공용 표시 (aos 없이 쓰는 헤더들이 함께 씀)
#pragma once
#if defined(__CUDACC__)
#define SV_HD __host__ __device__ __forceinline__
#define SV_HDN static __host__ __device__ __noinline__  // 큰 함수: 장치 코드에서 펼치지 않음(컴파일 시간·레지스터)
#else
#define SV_HD inline
#define SV_HDN inline
#endif
