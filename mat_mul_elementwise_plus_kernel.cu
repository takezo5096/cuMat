#include "mat_mul_elementwise_plus_kernel.h"

#define BLOCK_SIZE 16

__global__ void mat_mul_elementwise_plus_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, float alpha, float beta, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] += alpha * src1[row * n + col] * beta * src2[row * n + col];
    }

}

void mat_mul_elementwise_plus_kernel_exec(const float *src1, const float *src2, float *dst, float alpha, float beta, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_mul_elementwise_plus_kernel<<<grid, block>>>(src1, src2, dst, alpha, beta, m, n);
    cudaThreadSynchronize();
}
