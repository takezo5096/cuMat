#include "mat_sqrt_d_kernel.h"

#define BLOCK_SIZE 32

__device__ __forceinline__ float mat_sqrt_d (float a, float alpha){
    //return 0.5 * pow(a + alpha, -0.5f);
    return 0.5 * 1.0/sqrt(a + alpha);
}


__global__ void mat_sqrt_d_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float alpha){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = mat_sqrt_d(src[row * n + col], alpha);
    }
}

void mat_sqrt_d_kernel_exec(const float *src, float *dst, int m, int n, float alpha){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_sqrt_d_kernel<<<grid, block>>>(src, dst, m, n, alpha);
    cudaThreadSynchronize();

}
