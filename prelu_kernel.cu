#include "prelu_kernel.h"

#define BLOCK_SIZE 32

__device__ __forceinline__ float prelu(float x, float a){
    return x > 0.0f ? x : a * x;
}

__global__ void prelu_kernel (const float * __restrict__ src, const float * __restrict__ a,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){

        int idx = row * n + col;
        dst[idx] = prelu(src[idx], a[idx]);
    }

}

void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    prelu_kernel<<<grid, block>>>(src, a, dst, m, n);
    cudaThreadSynchronize();
}
