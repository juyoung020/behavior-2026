// DLPack 최소 정의 (DLPack v0.8 의 DLManagedTensor, ABI 그대로). JAX 의 jax.dlpack.from_dlpack 가 이것을 받는다.
// 원 헤더(dmlc/dlpack, Apache-2.0)의 구조체 배치와 같아야 한다 — 필드 순서·크기를 바꾸지 말 것.
#pragma once
#include <cstdint>

extern "C" {

enum { kDLCPU = 1, kDLCUDA = 2 };
enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBool = 6 };

typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;  // NULL = 연속(C 순서)
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;
}
