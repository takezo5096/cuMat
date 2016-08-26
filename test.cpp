#include <iostream>

#include "cuMat.h"

MallocCounter mallocCounter;

int main(){

    cuMat m1(3, 5), m2(3, 5), m3(3, 5), r(3, 5);

    m1.fill(1);
    m2.fill(2);
    m3.fill(3);


    std::cout << "m1:" << endl << m1;
    std::cout << "m2:" << endl << m2;
    m2.memSetHost(0, 0, -4);
    m2.memHostToDevice();
    std::cout << "m2:" << endl << m2;
/*
    m1.mul_plus(m2, m1, 1.0, 1.0);
    std::cout << "m1:" << endl << m1;

    float l2v = m1.l2();
    std::cout << l2v << std::endl;

    float sum = m1.sum();
    std::cout << sum << std::endl;

    cuMat rt = m1.tanh();
    std::cout << rt << std::endl;

    cuMat rt_d = m1.tanh_d();
    std::cout << rt_d << std::endl;

    cuMat sg = m1.sigmoid();
    std::cout << sg << std::endl;

    cuMat sg_d = m1.sigmoid_d();
    std::cout << sg_d << std::endl;

    cuMat relu = m1.relu();
    std::cout << relu << std::endl;

    cuMat relu_d = m1.relu_d();
    std::cout << relu_d << std::endl;

    cuMat sq = m1.sqrt();
    std::cout << sq << std::endl;

    m1.mul_plus(m2, m1, 1.0, 1.0);
    std::cout << m1 << std::endl;

    m2.dropout(m2, m3, 0.8);
    std::cout << m2 << std::endl;
    std::cout << m3 << std::endl;

    m1.adam2(m2, m3, r, 0.9, 0.99, 0.1, 1e-8);
    std::cout << r << std::endl;
*/
    m2 = m2.softmax();
    std::cout << "m2 softmax:" << endl << m2;

    m2.softmax_cross_entropy(m1, r);
    std::cout << "m2 softmax_cross_entropy:" << endl << r;

/*
    float sum = m1.sum();
    std::cout << sum << std::endl;

    cuMat relu = m2.relu();
    std::cout << relu << std::endl;
*/
    return 1;
}
