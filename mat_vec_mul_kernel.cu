#include "mat_vec_mul_kernel.h"

#define BLOCK_SIZE 32

__global__ void mat_vec_mul_kernel (const float * __restrict__ src_mat,
                                    const float * __restrict__ src_vec,
                                float * __restrict__ dst, int m, int n, int axis){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){

        if (axis == 0) dst[row * n + col] = src_mat[row * n + col] * src_vec[col];
        if (axis == 1) dst[row * n + col] = src_mat[row * n + col] * src_vec[row];
    }

}

void mat_vec_mul_kernel_exec(const float *src_mat, const float *src_vec, float *dst, int m, int n, int axis){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_vec_mul_kernel<<<grid, block>>>(src_mat, src_vec, dst, m, n, axis);
    cudaThreadSynchronize();

}
