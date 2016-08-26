#include "mat_l2_kernel.h"

#define BLOCK_SIZE 16

#include <stdio.h>

/*
 *  * 行列要素の合計を計算するカーネル
 *   */
__global__ void mat_l2_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        atomicAdd(dst, src[row * n + col]*src[row * n + col]);
    }
}


void mat_l2_kernel_exec(const float *src, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_l2_kernel<<<grid, block>>>(src, dst, m, n);
    cudaThreadSynchronize();

}
