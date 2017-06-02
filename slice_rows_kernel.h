#include <cuda_runtime.h>

#ifndef _slice_rows_kernel_
#define _slice_rows_kernel_


__global__ void slice_rows_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, int offset, int len);
__global__ void join_rows_kernel (const float * __restrict__ src,
                                   float * __restrict__ dst, int m, int n, int offset, int len);
#ifdef __cplusplus
extern "C" {
#endif
    void slice_rows_kernel_exec(const float *src, float *dst, int m, int n, int offset, int len);
    void join_rows_kernel_exec(const float *src, float *dst, int m, int n, int offset, int len);
#ifdef __cplusplus
};
#endif

#endif
