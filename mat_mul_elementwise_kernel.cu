#include "mat_mul_elementwise_kernel.h"

#define BLOCK_SIZE 16


__global__ void mat_mul_elementwise_kernel (const float * __restrict__ src1,
                                const float * __restrict__ src2,
                                float * __restrict__ dst, const int m, const int n){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){
        dst[row * n + col] = src1[row * n + col] * src2[row * n + col];
    }
}


void mat_mul_elementwise_kernel_exec(const float *src1, const float *src2, float *dst, const int m, const int n){

    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    /* lunch kernel */
    mat_mul_elementwise_kernel<<<grid, block>>>(src1, src2, dst, m, n);
    cudaThreadSynchronize();
}




/*
#define MATRIX_SIZE_M 5
#define MATRIX_SIZE_N 3

int main(int argc, char** argv){

    const int m = MATRIX_SIZE_M;
    const int n = MATRIX_SIZE_N;

    int matrixSize = sizeof(unsigned int) * m * n;

    //* malloc host memory
    float *hMatrixA = (float *)malloc(matrixSize);
    float *hMatrixB = (float *)malloc(matrixSize);
    float *hMatrixC = (float *)malloc(matrixSize);

    //* init data
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            hMatrixA[i * n + j] = i*n+j;
            hMatrixB[i * n + j] = i*n+j;
            printf("%f ", hMatrixA[i * n + j]);
        }
        printf("\n");
    }

    printf("------------------------------\n");

    //* variables for device
    float* dMatrixA;
    float* dMatrixB;
    float* dMatrixC;

    //* malloc device memory
    cudaMalloc((void**)&dMatrixA, matrixSize);
    cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dMatrixB, matrixSize);
    cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dMatrixC, matrixSize);

    mat_mul_elementwise_kernel_exec(dMatrixA, dMatrixB, dMatrixC, m, n);


    //* copy device memory of the result to host one
    cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost);

    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            printf("%f ", hMatrixC[i * n + j]);
        }
        printf("\n");
    }


    //*free host and device memories
    free(hMatrixA);
    free(hMatrixB);
    free(hMatrixC);
    cudaFree(dMatrixA);
    cudaFree(dMatrixB);
    cudaFree(dMatrixC);

    //* exit
    cudaThreadExit();
    return 0;
}
*/
