
#include <cuda_runtime.h>

#ifndef _batch_sum_kernel_
#define _batch_sum_kernel_

/*
 *  * バッチの合計を計算するカーネル
 *   */
__global__ void batch_sum_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void batch_sum_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
