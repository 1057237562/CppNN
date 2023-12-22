#ifndef LAYER_CPP
#define LAYER_CPP

#include "mutil.cpp"
#include "optimizer.cpp"

using namespace mutil;

class Layer {

public:
    virtual Mat forward(Mat& in) = 0;
    virtual Mat backward(Mat& in) = 0;
    virtual void randomize(default_random_engine& e) = 0;
    virtual void learn(Optimizer* optimizer) = 0;
};

class FlattenLayer : public Layer {
public:
    int h, w;
    FlattenLayer(int h, int w)
    {
        this->h = h;
        this->w = w;
    }
    Mat forward(Mat& in)
    {
        Mat ret(1, h * w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                ret[0][i * w + j] = in[i][j];
            }
        }
        return ret;
    }
    Mat backward(Mat& in)
    {
        Mat ret(h, w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                ret[i][j] = in[0][i * w + j];
            }
        }
        return ret;
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }
};

class LinearLayer : public Layer {
protected:
    Mat x, y;

public:
    int in, out;
    LinearLayer(int in, int out)
        : x(1, in)
        , y(1, out)
    {
        this->in = in;
        this->out = out;
    }
    virtual Mat forward(Mat& in) = 0;
    virtual Mat backward(Mat& in) = 0;
    virtual void randomize(default_random_engine& e) = 0;
    virtual void learn(Optimizer* optimizer) = 0;
};

class FullyConnectedLayer : public LinearLayer {
protected:
    Mat w, b;

public:
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;
    FullyConnectedLayer(int in, int out)
        : LinearLayer(in, out)
        , w(in, out)
        , b(1, out)
        , delta_w(in, out)
        , delta_b(1, out)
        , nabla_w(in, out)
        , nabla_b(1, out)
    {
        nabla_w.clear(), nabla_b.clear();
    }
    Mat forward(Mat& in)
    {
        Mat ret = in * w + b;
        x = in;
        y = ret;
        return ret;
    }
    Mat backward(Mat& in)
    {
        delta_w = x.transpose() * in;
        delta_b = in;
        nabla_w += delta_w;
        nabla_b += delta_b;
        Mat wT = w.transpose();
        return in * wT;
    }

    void randomize(default_random_engine& e)
    {
        w.randomize(e), b.randomize(e);
    }

    void learn(Optimizer* optimizer)
    {
        w = optimizer->optimize(w, nabla_w);
        b = optimizer->optimize(b, nabla_b);
        nabla_w.clear();
        nabla_b.clear();
    }
};

class SigmoidLayer : public LinearLayer {
public:
    SigmoidLayer(int in)
        : LinearLayer(in, in)
    {
    }
    Mat forward(Mat& in)
    {
        x = in;
        y = mutil::sigmoid(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        return in.dot(mutil::sigmoid_prime(x));
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }
};

class RELULayer : public LinearLayer {
public:
    RELULayer(int in)
        : LinearLayer(in, in)
    {
    }
    Mat forward(Mat& in)
    {
        x = in;
        y = mutil::relu(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        return in.dot(mutil::relu_prime(x));
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }
};

#endif