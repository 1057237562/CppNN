#ifndef LAYER_CPP
#define LAYER_CPP

#include "mutil.cpp"
#include "optimizer.cpp"

using namespace mutil;

class Layer
{

public:
    virtual Mat forward(Mat &in) = 0;
    virtual Mat backward(Mat &in) = 0;
    virtual void randomize(default_random_engine &e) = 0;
    virtual void learn(Optimizer *optimizer) = 0;
};

class FlattenLayer : public Layer
{
public:
    int h, w;
    FlattenLayer(int h, int w)
    {
        this->h = h;
        this->w = w;
    }
    Mat forward(Mat &in)
    {
        Mat ret(1, h * w);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                ret[0][i * w + j] = in[i][j];
            }
        }
        return ret;
    }
    Mat backward(Mat &in)
    {
        Mat ret(h, w);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                ret[i][j] = in[0][i * w + j];
            }
        }
        return ret;
    }

    void randomize(default_random_engine &e)
    {
    }

    void learn(Optimizer *optimizer)
    {
    }
};

class LinearLayer : public Layer
{
protected:
    Mat x, y;

public:
    int in, out;
    LinearLayer(int in, int out)
        : x(1, in), y(1, out)
    {
        this->in = in;
        this->out = out;
    }
    virtual Mat forward(Mat &in) = 0;
    virtual Mat backward(Mat &in) = 0;
    virtual void randomize(default_random_engine &e) = 0;
    virtual void learn(Optimizer *optimizer) = 0;
};

class DenseLayer : public LinearLayer
{
protected:
    Mat w, b;
    mutil::RandomMethod rmtd;

public:
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;
    DenseLayer(int in, int out, mutil::RandomMethod mtd = mutil::NORMAL)
        : LinearLayer(in, out), w(in, out), b(1, out), delta_w(in, out), delta_b(1, out), nabla_w(in, out), nabla_b(1, out), rmtd(mtd)
    {
        nabla_w.clear(), nabla_b.clear();
    }
    Mat forward(Mat &in)
    {
        Mat ret = in * w + b;
        x = in;
        y = ret;
        return ret;
    }
    Mat backward(Mat &in)
    {
        delta_w = x.transpose() * in;
        delta_b = in;
        nabla_w += delta_w;
        nabla_b += delta_b;
        Mat wT = w.transpose();
        return in * wT;
    }

    void randomize(default_random_engine &e)
    {
        w.randomize(e, rmtd), b.randomize(e, rmtd);
    }

    void learn(Optimizer *optimizer)
    {
        w = optimizer->optimize(w, nabla_w);
        b = optimizer->optimize(b, nabla_b);
        nabla_w.clear();
        nabla_b.clear();
    }
};

class SigmoidLayer : public Layer
{
protected:
    Mat x, y;

public:
    SigmoidLayer()
    {
    }
    Mat forward(Mat &in)
    {
        x = in;
        y = mutil::sigmoid(in);
        return in;
    }
    Mat backward(Mat &in)
    {
        return in.dot(mutil::sigmoid_prime(x));
    }

    void randomize(default_random_engine &e)
    {
    }

    void learn(Optimizer *optimizer)
    {
    }
};

class RELULayer : public Layer
{
protected:
    Mat x, y;

public:
    RELULayer()
    {
    }
    Mat forward(Mat &in)
    {
        x = in;
        y = mutil::relu(in);
        return in;
    }
    Mat backward(Mat &in)
    {
        return in.dot(mutil::relu_prime(x));
    }

    void randomize(default_random_engine &e)
    {
    }

    void learn(Optimizer *optimizer)
    {
    }
};

class ConvLayer : public Layer
{
protected:
    Mat x, y;
    mutil::RandomMethod rmtd;
    vector<int> in_size, kernel_size;

    int stride, padding;
    Mat w, b;
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;

public:
    ConvLayer(int height, int width, int channel, int kernel_height, int kernel_width, int kernel_count, int stride, int padding, mutil::RandomMethod mtd = NORMAL)
        : w(channel * kernel_count, kernel_height * kernel_width), b(channel * kernel_count, 1), delta_w(channel * kernel_count, kernel_height * kernel_width), delta_b(channel * kernel_count, 1), nabla_w(channel * kernel_count, kernel_height * kernel_width), nabla_b(channel * kernel_count, 1), rmtd(mtd)
    {
        this->in_size = {channel, height, width};
        this->kernel_size = {kernel_count, kernel_height, kernel_width};
        this->stride = stride;
        this->padding = padding;
        nabla_w.clear(), nabla_b.clear();
    }
    Mat forward(Mat &in)
    {
        Tensor tensor(in_size, in);
        pair<int, int> out_size = mutil::compute_output_size(in_size[1], in_size[2], kernel_size[1], kernel_size[2], stride, padding);
        Mat ret(kernel_size[0], out_size.first * out_size.second);
        ret.clear();
        for (int i = 0; i < in_size[0]; i++)
        {
            Kernel img(in_size[1], in_size[2], tensor[i]);
            for (int j = 0; j < kernel_size[0]; j++)
            {
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
    Mat backward(Mat &in)
    {
        Tensor img_tensor(in_size, x);
        Tensor delta_tensor(kernel_size, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        delta_w.clear();
        for (int i = 0; i < in_size[0]; i++)
        {
            Kernel img(in_size[1], in_size[2], img_tensor[i]);
            for (int j = 0; j < kernel_size[0]; j++)
            {
                Kernel dw_kernel(kernel_size[1], kernel_size[2], delta_w[i * kernel_size[0] + j]);
                Kernel in_kernel(kernel_size[1], kernel_size[2], delta_tensor[j]);
                mutil::conv(img, in_kernel, dw_kernel, stride, padding);
                delta_b[i * kernel_size[0] + j][0] = mutil::sum(in_kernel);
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::deconv(in_kernel, kernel, out, stride, padding);
            }
        }
        nabla_w += delta_w;
        nabla_b += delta_b;
        return ret;
    }

    void randomize(default_random_engine &e)
    {
        w.randomize(e, rmtd), b.randomize(e, rmtd);
    }

    void learn(Optimizer *optimizer)
    {
        w = optimizer->optimize(w, nabla_w);
        b = optimizer->optimize(b, nabla_b);
        nabla_w.clear();
        nabla_b.clear();
    }
};

class PoolingLayer : public Layer
{
    enum Type
    {
        MAX,
        MEAN
    };

private:
    pair<int, int> out_size;

protected:
    Mat x, y;
    vector<int> in_size;
    pair<int, int> pool_size;
    int stride;
    Type type;

public:
    PoolingLayer(int height, int width, int channel, pair<int, int> size, int stride, Type type = MAX) : x(channel, height * width), y(channel, height * width), pool_size(size), stride(stride), type(type)
    {
        in_size = {channel, height, width};
        pair<int, int> out_size = mutil::compute_output_size(in_size[1], in_size[2], size.first, size.second, stride, 0);
    }

    Mat forward(Mat &in)
    {
        Tensor tensor(in_size, in);
        Mat ret(in_size[0], out_size.first * out_size.second);
        ret.clear();
        if (type == MAX)
        {
            for (int i = 0; i < in_size[0]; i++)
            {
                Kernel img(in_size[1], in_size[2], tensor[i]);
                Kernel out(out_size.first, out_size.second, ret[i]);
                mutil::max_pooling(img, out, pool_size, stride);
            }
        }
        if (type == MEAN)
        {
            for (int i = 0; i < in_size[0]; i++)
            {
                Kernel img(in_size[1], in_size[2], tensor[i]);
                Kernel out(out_size.first, out_size.second, ret[i]);
                mutil::mean_pooling(img, out, pool_size, stride);
            }
        }
        x = in;
        y = ret;
        return ret;
    }

    Mat backward(Mat &in)
    {
        Tensor img_tensor(in_size, x);
        Tensor delta_tensor({in_size[0], out_size.first, out_size.second}, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        if (type == MAX)
        {
            for (int i = 0; i < in_size[0]; i++)
            {
                Kernel img(in_size[1], in_size[2], img_tensor[i]);
                Kernel delta(out_size.first, out_size.second, delta_tensor[i]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::max_pooling_prime(img, delta, out, pool_size, stride);
            }
        }
        if (type == MEAN)
        {
            for (int i = 0; i < in_size[0]; i++)
            {
                Kernel delta(out_size.first, out_size.second, delta_tensor[i]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::mean_pooling_prime(delta, out, pool_size, stride);
            }
        }
        return ret;
    }

    void randomize(default_random_engine &e)
    {
    }

    void learn(Optimizer *optimizer)
    {
    }
};

#endif