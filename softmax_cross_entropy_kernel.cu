#include "softmax_cross_entropy_kernel.h"

#define BLOCK_SIZE 16

/*
 * softmax cross entropy kernel
 * dst = -sigma(log(src1 + 1e-8)*src2)
 *   */
__global__ void softmax_cross_entropy_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = -1.0f * std::log(src1[row * n + col] + 1e-8) * src2[row * n + col];
    }
}

void softmax_cross_entropy_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    softmax_cross_entropy_kernel<<<grid, block>>>(src1, src2, dst, m, n);
    cudaThreadSynchronize();

}
