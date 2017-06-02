#include <iostream>

#include "cuMat.h"

MallocCounter mallocCounter;



int main(){


    float b[6] = {1,2,3,4,5,6};

    cuMat dm(3, 2);
    dm.memSetHost(b);
    cout << dm;
    cuMat dm_b = dm.batch_sum().vec_to_mat(10);
    cout << dm_b;
    exit(0);



    int image_size = 5;
    int channel_num = 3;
    int batch_size = 2;
    float a[image_size * image_size * channel_num * batch_size] = {
            1,   2,   3,   4,   5,
            6,   7,   8,   9,   10,
            11,  12,  13,  14,  15,
            16,  17,  18,  19,  20,
            21,  22,  23,  24,  25,
            101, 102, 103, 104, 105,
            106, 107, 108, 109, 110,
            111, 112, 113, 114, 115,
            116, 117, 118, 119, 120,
            121, 122, 123, 124, 125,
            201, 202, 203, 204, 205,
            206, 207, 208, 209, 210,
            211, 212, 213, 214, 215,
            216, 217, 218, 219, 220,
            221, 222, 223, 224, 225,

            1001,   1002,   1003,  1004,   1005,
            1006,   1007,   1008,   1009,   10010,
            1011,  1012,  1013,  1014,  1015,
            1016,  1017,  1018,  1019,  1020,
            1021,  1022,  1023,  1024,  1025,
            1101, 1102, 1103, 1104, 1105,
            1106, 1107, 1108, 1109, 1110,
            1111, 1112, 1113, 1114, 1115,
            1116, 1117, 1118, 1119, 1120,
            1121, 1122, 1123, 1124, 1125,
            1201, 1202, 1203, 1204, 1205,
            1206, 1207, 1208, 1209, 1210,
            1211, 1212, 1213, 1214, 1215,
            1216, 1217, 1218, 1219, 1220,
            1221, 1222, 1223, 1224, 1225
    };


    /**
        * Each dimension h and w of the output images is computed as followed:
        * outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride
        */

    cuMat am(image_size * image_size * channel_num, batch_size);
    am.memSetHost(a);


    cout << "am" << endl;
    cout << am;

    cuMat sam = am.sliceRows(image_size * image_size, image_size * image_size);
    cout << sam;

    am.joinRows(sam, 0, image_size * image_size);
    cout << am;

    exit(1);

    int i = 1;
    int data_index = i*(channel_num * image_size * image_size);
    float *one_m = am.mDevice + data_index;
    cuMat one_m_dev(image_size * image_size, channel_num);
    one_m_dev.memSetDevice(one_m);

    cout << one_m_dev;

    am.memSetDeviceCol(one_m_dev.mDevice, 0);
    cout << am;

    exit(1);

/*
    cuMat pooled = am.pooling(1, image_size, image_size, channel_num, 3, 3, 2, 2, 0, 0, 0, 0);
    cout << pooled;
    //exit(1);

    cuMat p_grad(4, 3);
    p_grad.ones();

    cuMat dxr = am.pooling_backward(1, p_grad.mDevice, image_size, image_size, channel_num, 3, 3, 2, 2, 0, 0, 0, 0);
    cout << dxr;
*/


    int filter_size = 3;
    int padding_size = 0;

    int outputDimW, outputDimH;

    cuMat bm = am.im2col(image_size, image_size, channel_num, filter_size, filter_size,
            1, 1, padding_size, padding_size, padding_size, padding_size, outputDimW, outputDimH);


    cout << "output dim:" << outputDimW << " x " << outputDimH << endl;

    cout << "bm" << endl;
    cout << bm;
    //cuMat bm_t = bm.transpose();
    //cout << bm_t;



    cuMat cm = bm.col2im(image_size, image_size, channel_num, filter_size, filter_size,
                         1, 1, padding_size, padding_size, padding_size, padding_size);
    cout << "cm" << endl;
    cout << cm;


    return 0;
}