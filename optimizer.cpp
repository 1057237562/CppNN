#ifndef OPTIMIZER_CPP
#define OPTIMIZER_CPP

#include "mutil.cpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using namespace mutil;

class Optimizer {

public:
    virtual Mat optimize(Mat& mat, Mat& nabla) = 0;
};

class SDG : public Optimizer {

    float learning_rate;

public:
    SDG(float learning_rate)
    {
        this->learning_rate = learning_rate;
    }

    Mat optimize(Mat& mat, Mat& nabla)
    {
        float multiplier = learning_rate;
        Mat ret = nabla * multiplier;
        return mat - ret;
    }
};

#endif