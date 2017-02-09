#include "mat_cos_kernel.h"

#define BLOCK_SIZE 32

__device__ __forceinline__ float mat_cos(float a, float alpha){
    return std::cos(a+alpha);
}

__global__ void mat_cos_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float alpha){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = mat_cos(src[row * n + col], alpha);
    }

}

void mat_cos_kernel_exec(const float *src, float *dst, int m, int n, float alpha){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_cos_kernel<<<grid, block>>>(src, dst, m, n, alpha);
    cudaThreadSynchronize();

}
