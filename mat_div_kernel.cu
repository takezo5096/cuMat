#include "mat_div_kernel.h"

#define BLOCK_SIZE 32

__global__ void mat_div_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = src1[row * n + col] / src2[row * n + col];
    }
}
void mat_div_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_div_kernel<<<grid, block>>>(src1, src2, dst, m, n);
    cudaThreadSynchronize();
}
