//
// Created by 藤田 毅 on 2016/12/02.
//

#include <cuda_runtime.h>

#ifndef _element_wise_clip_kernel_
#define _element_wise_clip_kernel_

__device__ __forceinline__ float element_wise_clip(float a, float threshold);

/*
 *  * シグモイドカーネル
 *   */
__global__ void element_wise_clip_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float threshold);
#ifdef __cplusplus
extern "C" {
#endif
void element_wise_clip_kernel_exec(const float *src, float *dst, int m, int n, float threshold);
#ifdef __cplusplus
};
#endif

#endif


