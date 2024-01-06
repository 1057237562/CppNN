#ifndef NETWORK

#define NETWORK

#include "layer.cpp"
#include "mutil.cpp"
#include "optimizer.cpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

#define endl '\n'

using namespace std;
using namespace mutil;

function<Mat(Mat&, Mat&)> MSE = [](Mat& res, Mat& ans) {
    return (res - ans) * (1.0f / ans.size.first / ans.size.second);
};

function<Mat(Mat&, Mat&)> L1 = [](Mat& res, Mat& ans) {
    Mat ret = Mat(res.size.first, res.size.second);
    for (int i = 0; i < res.size.first; i++) {
        for (int j = 0; j < res.size.second; j++) {
            ret[i][j] = res[i][j] > ans[i][j] ? 1 : -1; // Seem to be wrong at res == ans
                                                        // result will be concussion
        }
    }
    return ret;
};

function<Mat(Mat&, Mat&)> CrossEntropy = [](Mat& res, Mat& ans) {
    return (ans / res) * -1.0f;
};

class Network {

protected:
    vector<Layer*> layers;
    Optimizer* optimizer;
    int batch_size;

public:
    function<Mat(Mat&, Mat&)> costfunc = [](Mat& res, Mat& ans) {
        return (res - ans) * (1.0f / ans.size.first / ans.size.second);
    };
    int forwardTime = 0;
    int backwardTime = 0;
    Network(vector<Layer*> layers, Optimizer* optimizer, int batch_size)
        : batch_size(batch_size)
    {
        this->layers = layers;
        this->optimizer = optimizer;
    }

    void init(int seed)
    {
        default_random_engine e;
        e.seed(seed);
        for (auto layer : layers) {
            layer->randomize(e);
        }
    }

    void init()
    {
        default_random_engine e;
        for (auto layer : layers) {
            layer->randomize(e);
        }
    }

    Mat forward(Mat in)
    {
        auto start = clock();
        for (auto layer : layers) {
            in = layer->forward(in);
        }
        auto end = clock();
        forwardTime += end - start;
        return in;
    }

    void backPropagation(Mat result, Mat answer)
    {
        auto start = clock();
        Mat delta = costfunc(result, answer);
        for (int i = layers.size() - 1; i >= 0; i--) {
            delta = layers[i]->backward(delta);
        }
        auto end = clock();
        backwardTime += end - start;
    }

    void train(vector<pair<Mat, Mat>>& data, default_random_engine e = default_random_engine())
    {
        shuffle(data.begin(), data.end(), e);
        int index = 0;
        for (; index < data.size(); index++) {
            Mat result = forward(data[index].first);
            backPropagation(result, data[index].second);
            for (index++; index % batch_size && index < data.size(); index++) {
                Mat result = forward(data[index].first);
                backPropagation(result, data[index].second);
            }
            for (auto layer : layers) {
                layer->learn(optimizer);
            }
            if (index % (batch_size * 100) == 0)
                cout << "Processing Batches : " << ((index + 1) / batch_size) << "/"
                     << (data.size() / batch_size) << endl;
        }
    }

    void saveCheckpoint(ofstream& out)
    {
        for (auto layer : layers) {
            layer->saveCheckpoint(out);
        }
    }

    void loadCheckpoint(ifstream& in)
    {
        for (auto layer : layers) {
            layer->loadCheckpoint(in);
        }
    }
};

#endif