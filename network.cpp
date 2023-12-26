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

#define endl '\n'

using namespace std;
using namespace mutil;

class Network {

protected:
    vector<Layer*> layers;
    Optimizer* optimizer;

public:
    function<Mat(Mat, Mat)> costfunc = [&](Mat res, Mat ans) {
        return res - ans;
    };
    int forwardTime = 0;
    int backwardTime = 0;
    Network(vector<Layer*> layers, Optimizer* optimizer)
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

    void train()
    {
        optimizer->shuffle();
        for (auto data = optimizer->next(); !optimizer->end();) {
            for (data = optimizer->next(); optimizer->hasNext(); data = optimizer->next()) {
                Mat result = forward(data.first);
                backPropagation(result, data.second);
            }
            for (auto layer : layers) {
                layer->learn(optimizer);
            }
            if (optimizer->index % (optimizer->batch_size * 100) == 0)
                cout << "Processing Batches : " << ((optimizer->index + 1) / optimizer->batch_size) << "/"
                     << (optimizer->count() / optimizer->batch_size) << endl;
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