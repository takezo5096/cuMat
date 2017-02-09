#include "tanh_kernel.h"

#define BLOCK_SIZE 32

__device__ __forceinline__ float tanh_f(float a){
    return std::tanh(a);
}

__global__ void tanh_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = tanh_f(src[row * n + col]);
    }

}
void tanh_kernel_exec(const float *src, float *dst, int m, int n){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    tanh_kernel<<<grid, block>>>(src, dst, m, n);
    cudaThreadSynchronize();
}
