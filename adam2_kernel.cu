#include "adam2_kernel.h"
#include <curand_kernel.h>

#define BLOCK_SIZE 32

//mm += (1.0 - beta1) * (mg - mm);
//mv += (1.0 - beta2) * (mg*mg - mv);
//dst = lr * mm / (mv.sqrt() + 1e-8);

__global__ void adam2_kernel (
                                                float * __restrict__ mm,
                                                float * __restrict__ mv,
                                                const float * __restrict__ mg,
                                                float * __restrict__ dst,
                                                float beta1, float beta2,
                                                float lr, float e, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        int idx = row * n + col;
        mm[idx] += (1.0f - beta1) * (mg[idx] - mm[idx]);
        mv[idx] += (1.0f - beta2) * (mg[idx]*mg[idx] - mv[idx]);
        dst[idx] = lr * mm[idx] / (std::sqrt(mv[idx]) + e);
    }
}

void adam2_kernel_exec(float *mm, float *mv, const float *mg, float *dst, float beta1, float beta2, float lr, float e, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    adam2_kernel<<<grid, block>>>(mm, mv, mg, dst, beta1, beta2, lr, e, m, n);
    cudaThreadSynchronize();

}
