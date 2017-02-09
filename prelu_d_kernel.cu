#include "prelu_d_kernel.h"

#define BLOCK_SIZE 32


__global__ void prelu_d_kernel (const float * __restrict__ src, const float * __restrict__ a,
                                float * __restrict__ dst, float * __restrict__ da, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        int idx = row * n + col;

        dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
        da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
    }

}

void prelu_d_kernel_exec(const float *src, const float *a, float *dst, float *da, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    prelu_d_kernel<<<grid, block>>>(src, a, dst, da, m, n);
    cudaThreadSynchronize();
}
