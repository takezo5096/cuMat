#include <cuda_runtime.h>

#ifndef _prelu_d_kernel_
#define _prelu_d_kernel_


#ifdef __cplusplus
extern "C" {
#endif
    void prelu_d_kernel_exec(const float *src, const float *a, float *dst, float *da, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
