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

class DenseLayer : public LinearLayer {
protected:
    Mat w, b;

public:
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;
    DenseLayer(int in, int out)
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

class SigmoidLayer : public Layer {
protected:
    Mat x, y;

public:
    SigmoidLayer()
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

class RELULayer : public Layer {
protected:
    Mat x, y;

public:
    RELULayer()
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

class ConvLayer : public Layer {
protected:
    Mat x, y;
    vector<int> in_size, kernel_size;

    int stride, padding;
    Mat w, b;
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;

public:
    ConvLayer(int height, int width, int channel, int kernel_height, int kernel_width, int kernel_count, int stride, int padding)
        : w(channel*kernel_count, kernel_height * kernel_width)
        , b(1, channel*kernel_count)
        , delta_w(channel*kernel_count, kernel_height * kernel_width)
        , delta_b(1, channel*kernel_count)
        , nabla_w(channel*kernel_count, kernel_height * kernel_width)
        , nabla_b(1, channel*kernel_count)
    {
        this->in_size = {channel, height, width };
        this->kernel_size = { kernel_count,kernel_height, kernel_width };
        this->stride = stride;
        this->padding = padding;
        nabla_w.clear(), nabla_b.clear();
    }
    Mat forward(Mat& in)
    {
        Tensor tensor(in_size, in);
        pair<int,int> out_size = mutil::compute_output_size(in_size[1], in_size[2], kernel_size[1], kernel_size[2], stride, padding);
        Mat ret(kernel_size[0], out_size.first * out_size.second);
        ret.clear();
        for(int i = 0; i < in_size[0]; i++) {
            Kernel img(in_size[1], in_size[2], tensor[i]);
            for(int j = 0; j < kernel_size[0]; j++) {
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(out_size.first, out_size.second, ret[j]);
                mutil::conv(img, kernel, out, stride, padding);
                out += b[0][i * kernel_size[0] + j];
            }
        }
        x = in;
        y = ret;
        return ret;
    }
    Mat backward(Mat& in)
    {
        Tensor tensor(in_size, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        delta_w.clear();
        delta_b.clear();
        for(int i = 0; i < in_size[0]; i++) {
            Kernel img(in_size[1], in_size[2], tensor[i]);
            for(int j = 0; j < kernel_size[0]; j++) {
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::conv(img, kernel, out, stride, padding);
                out += b[0][i * kernel_size[0] + j];
            }
        }
        for(int i = 0; i < in_size[0]; i++) {
            Kernel img(in_size[1], in_size[2], tensor[i]);
            for(int j = 0; j < kernel_size[0]; j++) {
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(out_size.first, out_size.second, ret[i]);
                mutil::conv(img, kernel, out, stride, padding);
                out += b[0][i * kernel_size[0] + j];
                Kernel delta(kernel_size[1], kernel_size[2], delta_w[i * kernel_size[0] + j]);
                mutil::conv(img, out, stride, padding);
                delta_b[0][i * kernel_size[0] + j] += mutil::sum(out);
            }
        }
        nabla_w += delta_w;
        nabla_b += delta_b;
        return ret;
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

#endif