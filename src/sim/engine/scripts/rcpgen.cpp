// 이 CPU 의 rcpps / rsqrtps 표(가수 상위 12 비트)를 뽑고, 표 흉내가 float 전 범위에서 하드웨어와 같은지 전수 비교한다.
#include <xmmintrin.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>
static float rcp_hw(float x){ return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))); }
static float rsq_hw(float x){ return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
static uint32_t B(float f){ uint32_t u; memcpy(&u,&f,4); return u; }
static float F(uint32_t u){ float f; memcpy(&f,&u,4); return f; }
static uint32_t RCP[4096], RSQ0[4096], RSQ1[4096];  // 결과 비트 (x = 1.m, 2.m 기준)
static float rcp_emu(float x){
  uint32_t u=B(x), s=u&0x80000000u, e=(u>>23)&0xff, m=u&0x7fffff;
  if(e==0xff) return m? F(u|0x400000u) : F(s);            // NaN(조용한 NaN 으로) / inf -> 0
  if(e==0) return F(s|0x7f800000u);                         // 0 과 비정규 -> inf
  uint32_t r=RCP[m>>11];                                     // x in [1,2) 의 결과: 지수 126 또는 127
  int re=int((r>>23)&0xff) - (int(e)-127);
  if(re>=0xff) return F(s|0x7f800000u);
  if(re<=0) return F(s);                                     // 비정규 결과 -> 0 (추정, 전수 비교로 확인)
  return F(s | (uint32_t(re)<<23) | (r&0x7fffff));
}
static float rsq_emu(float x){
  uint32_t u=B(x), s=u&0x80000000u, e=(u>>23)&0xff, m=u&0x7fffff;
  if(e==0xff){ if(m) return F(u|0x400000u); return s? F(0xffc00000u) : F(0); }
  if(e==0) return F(s|0x7f800000u);
  if(s) return F(0xffc00000u);                               // 음수 -> NaN
  int E=int(e)-127;
  uint32_t r; int k;
  if((E&1)==0){ r=RSQ0[m>>11]; k=E/2; } else { r=RSQ1[m>>11]; k=(E-1)/2; }
  int re=int((r>>23)&0xff)-k;
  return F((uint32_t(re)<<23)|(r&0x7fffff));
}
int main(int argc,char**argv){
  for(uint32_t i=0;i<4096;i++){ RCP[i]=B(rcp_hw(F((127u<<23)|(i<<11)))); RSQ0[i]=B(rsq_hw(F((127u<<23)|(i<<11)))); RSQ1[i]=B(rsq_hw(F((128u<<23)|(i<<11)))); }
  for(int daz=0; daz<2; daz++){
    unsigned old=_mm_getcsr();
    if(daz) _mm_setcsr(_MM_MASK_MASK|_MM_FLUSH_ZERO_ON|(1<<6));
    std::atomic<uint64_t> br{0}, bs{0}; std::atomic<int64_t> fr{-1}, fs{-1};
    std::vector<std::thread> th; unsigned nt=std::thread::hardware_concurrency();
    for(unsigned t=0;t<nt;t++) th.emplace_back([&,t]{
      if(daz) _mm_setcsr(_MM_MASK_MASK|_MM_FLUSH_ZERO_ON|(1<<6));
      uint64_t lo=(1ull<<32)*t/nt, hi=(1ull<<32)*(t+1)/nt;
      for(uint64_t v=lo; v<hi; v++){ float x=F(uint32_t(v));
        uint32_t a=B(rcp_hw(x)), b=B(rcp_emu(x)); if(a!=b){ if(br++==0) fr=int64_t(v);} 
        uint32_t c=B(rsq_hw(x)), d=B(rsq_emu(x)); if(c!=d){ if(bs++==0) fs=int64_t(v);} }
    });
    for(auto&x:th) x.join();
    _mm_setcsr(old);
    printf("DAZ/FTZ=%d: rcp 다름 %llu (첫 0x%08llx hw 0x%08x emu 0x%08x), rsqrt 다름 %llu (첫 0x%08llx hw 0x%08x emu 0x%08x)\n", daz,
      (unsigned long long)br.load(), (unsigned long long)fr.load(), fr>=0?B(rcp_hw(F(uint32_t(fr)))):0, fr>=0?B(rcp_emu(F(uint32_t(fr)))):0,
      (unsigned long long)bs.load(), (unsigned long long)fs.load(), fs>=0?B(rsq_hw(F(uint32_t(fs)))):0, fs>=0?B(rsq_emu(F(uint32_t(fs)))):0);
  }
  if(argc>1){ FILE*f=fopen(argv[1],"w");
    fprintf(f,"// 자동 생성 (rcpgen): CPU 의 rcpps/rsqrtps 결과 비트, 입력 가수 상위 12 비트별. RCP: x=[1,2), RSQ0: x=[1,2), RSQ1: x=[2,4)\n");
    const char* nm[3]={"kRcpTable","kRsqTable0","kRsqTable1"}; uint32_t* tb[3]={RCP,RSQ0,RSQ1};
    for(int t=0;t<3;t++){ fprintf(f,"ENG_APPROX_TABLE(%s) = {\n",nm[t]); for(int i=0;i<4096;i++) fprintf(f,"0x%08xu,%s",tb[t][i],(i%8==7)?"\n":" "); fprintf(f,"};\n"); }
    fclose(f); }
  return 0;
}
