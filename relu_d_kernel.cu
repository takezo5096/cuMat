#include "relu_d_kernel.h"

#define BLOCK_SIZE 16

__device__ __forceinline__ float relu_d(float a){
    return a > 0.0f ? 1.0f:0.0f;
}

__global__ void relu_d_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = relu_d(src[row * n + col]);
    }

}

void relu_d_kernel_exec(const float *src, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    relu_d_kernel<<<grid, block>>>(src, dst, m, n);
    cudaThreadSynchronize();
}
