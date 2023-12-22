#ifndef OPTIMIZER_CPP
#define OPTIMIZER_CPP

#include "mutil.cpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace mutil;

class Optimizer {

public:
    int index = 0;
    virtual Mat optimize(Mat& mat, Mat& nabla) = 0;
    virtual pair<Mat, Mat>& next() = 0;
    virtual bool hasNext() = 0;
    virtual bool end() = 0;
    virtual void shuffle() = 0;
    virtual int count() = 0;
};

class SDG : public Optimizer {

    float learning_rate;
    int batch_size;
    vector<pair<Mat, Mat>> training_data;

public:
    SDG(vector<pair<Mat, Mat>>& training_data, float learning_rate, int batch_size)
    {
        this->learning_rate = learning_rate;
        this->batch_size = batch_size;
        this->training_data = training_data;
    }

    Mat optimize(Mat& mat, Mat& nabla)
    {
        float multiplier = learning_rate / batch_size;
        Mat ret = nabla * multiplier;
        return mat - ret;
    }

    pair<Mat, Mat>& next()
    {
        if (index == training_data.size() - 1) {
            index = 0;
        }
        index++;
        return training_data[index];
    }

    bool hasNext()
    {
        return index % batch_size != 0 && index != training_data.size() - 1;
    }

    bool end()
    {
        return index == training_data.size() - 1;
    }

    void shuffle()
    {
        std::random_shuffle(training_data.begin(), training_data.end());
    }

    int count()
    {
        return training_data.size();
    }
};

#endif