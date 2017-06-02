#include "slice_rows_kernel.h"
#include <stdio.h>
#define BLOCK_SIZE 32


__global__ void slice_rows_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, int offset, int len){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    //printf("%d x %d\n", row, col);
    if (offset <= row && row < offset+len && row < n && col < m){
        //printf("offset:%d len:%d %dx%d [%d] %f\n", offset, len, row, col, col * n + row, src[col * n + row]);
        dst[col * len + row-offset] = src[col * n + row];
    }
}

void slice_rows_kernel_exec(const float *src, float *dst, int m, int n, int offset, int len){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m+block.x-1)/block.x, (n+block.y-1)/block.y);

    //printf("m:%d n:%d offset:%d len:%d\n", m, n, offset, len);

    /* lunch kernel */
    slice_rows_kernel<<<grid, block>>>(src, dst, m, n, offset, len);
    cudaThreadSynchronize();

}


__global__ void join_rows_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, int offset, int len){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    //printf("%d x %d\n", row, col);
    if (offset <= row && row < offset+len && row < n && col < m){
        //printf("offset:%d len:%d %dx%d [%d] %f\n", offset, len, row, col, col * n + row, src[col * n + row]);
        dst[col * n + row] = src[col * len + row-offset];
    }
}

void join_rows_kernel_exec(const float *src, float *dst, int m, int n, int offset, int len){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m+block.x-1)/block.x, (n+block.y-1)/block.y);

    //printf("m:%d n:%d offset:%d len:%d\n", m, n, offset, len);

    /* lunch kernel */
    join_rows_kernel<<<grid, block>>>(src, dst, m, n, offset, len);
    cudaThreadSynchronize();

}
