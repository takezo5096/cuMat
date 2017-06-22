#ifndef _im2col_h_
#define _im2col_h_


void im2col_ongpu(float *im,
                  int channels, int height, int width,
                  int ksize, int stride, int pad, float *data_col);



void col2im_ongpu(float *data_col,
                  int channels, int height, int width,
                  int ksize, int stride, int pad, float *data_im);
#endif


