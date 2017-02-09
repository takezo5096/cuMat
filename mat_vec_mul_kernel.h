#include <cuda_runtime.h>

#ifndef _mat_vec_mul_kernel_
#define _mat_vec_mul_kernel_


/*
 *  * シグモイドカーネル
 *   */
__global__ void mat_vec_mul_kernel (const float * __restrict__ src_mat,
                                    const float * __restrict__ src_vec,
                                    float * __restrict__ dst, int m, int n, int axis);
#ifdef __cplusplus
extern "C" {
#endif
void mat_vec_mul_kernel_exec(const float *src_mat, const float *src_vec, float *dst, int m, int n, int axis);
#ifdef __cplusplus
};
#endif

#endif
