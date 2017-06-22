#ifndef _POOLING_H_
#define _POOLING_H_


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
                 size_t padBottom);


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
                         size_t padBottom);
#endif