#include <cuda_runtime.h>

#ifndef _prelu_kernel_
#define _prelu_kernel_


#ifdef __cplusplus
extern "C" {
#endif
    void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
