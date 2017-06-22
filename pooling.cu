
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include <iostream>

#define NUM_THREADS 1024


inline int divideUpwards(int a, int b) {   return (a + b - 1) / b ; }



template<typename T>
__global__ void maxPooling_gpu_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int windowWidth,
 const int windowHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {

    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T bestValue = data[y1 * width + x1] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        bestValue = max(bestValue, data[y * width + x]) ;
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}


template<typename T>
void pooling_gpu(T* pooled,
                 T const* data,
                 size_t width,
                 size_t height,
                 size_t depth,
                 size_t windowWidth,
                 size_t windowHeight,
                 size_t strideX,
                 size_t strideY,
                 size_t padLeft,
                 size_t padRight,
                 size_t padTop,
                 size_t padBottom)
{
  int pooledWidth = (width + (padLeft+padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - windowHeight)/strideY + 1 ;
  int pooledVolume = pooledWidth * pooledHeight * depth ;

      maxPooling_gpu_kernel<T>
      <<< divideUpwards(pooledVolume, NUM_THREADS), NUM_THREADS >>>
      (pooled, data,
       pooledWidth, pooledHeight, pooledVolume,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop);
      if (cudaGetLastError() != cudaSuccess) {
        std::cout
        <<"maxPooling_gpu_kernel error ("
        <<cudaGetErrorString(cudaGetLastError())
        <<")"<<std::endl ;
      }
}

template
void pooling_gpu<float>(float* pooled,
                        float const* data,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t windowWidth,
                        size_t windowHeight,
                        size_t strideX,
                        size_t strideY,
                        size_t padLeft,
                        size_t padRight,
                        size_t padTop,
                        size_t padBottom) ;




template<typename T>
__global__ void maxPoolingBackward_gpu_kernel
(T* dzdx,
 const T* data,
 const T* dzdy,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int windowWidth,
 const int windowHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {

    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;
    dzdx += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        int index = y * width + x ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }

    atomicAdd(dzdx + bestIndex, dzdy[pooledIndex]) ;
  }
}


template<typename T>
void poolingBackward_gpu(T* dzdx,
                         T const* data,
                         T const* dzdy,
                         size_t width,
                         size_t height,
                         size_t depth,
                         size_t windowWidth,
                         size_t windowHeight,
                         size_t strideX,
                         size_t strideY,
                         size_t padLeft,
                         size_t padRight,
                         size_t padTop,
                         size_t padBottom)
{
  int pooledWidth = (width + (padLeft+padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - windowHeight)/strideY + 1 ;
  int nthreads;

      nthreads = pooledWidth * pooledHeight * depth ;
      maxPoolingBackward_gpu_kernel<T>
      <<< divideUpwards(nthreads, NUM_THREADS), NUM_THREADS >>>
      (dzdx,
       data, dzdy,
       pooledWidth, pooledHeight, nthreads,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop);
      if (cudaGetLastError() != cudaSuccess) {
        std::cout
        <<"maxPooling_gpu_kernel error ("
        <<cudaGetErrorString(cudaGetLastError())
        <<")"<<std::endl ;
      }
}

template
void poolingBackward_gpu<float>(float* dzdx,
                                float const* data,
                                float const* dzdy,
                                size_t width,
                                size_t height,
                                size_t depth,
                                size_t windowWidth,
                                size_t windowHeight,
                                size_t strideX,
                                size_t strideY,
                                size_t padLeft,
                                size_t padRight,
                                size_t padTop,
                                size_t padBottom) ;

