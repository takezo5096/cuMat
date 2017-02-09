#include "element_wise_clip_kernel.h"

#define BLOCK_SIZE 32

__device__ __forceinline__ float element_wise_clip(float a, float threshold){

    float _a = std::abs(a);
    if (threshold < _a) return threshold/_a * a;
    else return a;
}

__global__ void element_wise_clip_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float threshold){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = element_wise_clip(src[row * n + col], threshold);
    }

}

void element_wise_clip_kernel_exec(const float *src, float *dst, int m, int n, float threshold){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    element_wise_clip_kernel<<<grid, block>>>(src, dst, m, n, threshold);
    cudaThreadSynchronize();

}
