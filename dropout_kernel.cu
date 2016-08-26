#include "dropout_kernel.h"
#include <curand_kernel.h>

#define BLOCK_SIZE 16

__device__ int WangHash(int a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__global__ void dropout_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, float * __restrict__ dst_idx, int m, int n, float p){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){

        // curand_init is very slow.
        // so we use the technique as bellow.
        // http://richiesams.blogspot.jp/2015/03/creating-randomness-and-acummulating.html
        // or https://devtalk.nvidia.com/default/topic/480586/curand-initialization-time/
        //generate random number
        int SEED = WangHash(1234);
        curandState_t state;
        int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        //curand_init( (SEED << 20) + threadId, 0, 0, &state);
        curand_init( SEED + threadId, 0, 0, &state);
        float randNum = curand_uniform(&state);


        if (randNum > p){
            dst[row * n + col] = src[row * n + col];
            dst_idx[row * n + col] = 1.0f;
        }
        else{
            dst[row * n + col] = 0.0f;
            dst_idx[row * n + col] = 0.0f;
        }

    }

}

void dropout_kernel_exec(const float *src, float *dst, float *dst_idx, int m, int n, float p){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    dropout_kernel<<<grid, block>>>(src, dst, dst_idx, m, n, p);
    cudaThreadSynchronize();
}
