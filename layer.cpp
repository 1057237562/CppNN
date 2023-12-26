#ifndef LAYER_CPP
#define LAYER_CPP

#include "debug.cpp"
#include "mutil.cpp"
#include "optimizer.cpp"
#include <fstream>
#include <mutex>
#include <thread>

using namespace mutil;

std::mutex tasks_mutex;
unsigned int threadCnt = 0;

static init::Initializer* getInit(init::Type type, int n)
{
    if (type == init::KAIMING)
        return new init::KaimingInit(n);
    if (type == init::XAVIER)
        return new init::XavierInit(n);
    if (type == init::UNIFORM)
        return new init::UniformInit();
    if (type == init::NORMAL)
        return new init::NormalInit();
    return nullptr;
}

class Layer {

public:
    virtual Mat& forward(Mat& in) = 0;
    virtual Mat backward(Mat& in) = 0;
    virtual void randomize(default_random_engine& e) = 0;
    virtual void learn(Optimizer* optimizer) = 0;
    virtual void saveCheckpoint(ofstream& ofstream) = 0;
    virtual void loadCheckpoint(ifstream& ifstream) = 0;
};

class FlattenLayer : public Layer {
private:
    pair<int, int> in_size;

public:
    FlattenLayer()
    {
    }
    Mat& forward(Mat& in)
    {
        in_size = in.size;
        in.size = { 1, in.size.first * in.size.second };
        return in;
    }
    Mat backward(Mat& in)
    {
        Mat ret(in_size.first, in_size.second);
        for (int i = 0; i < in_size.first; i++) {
            for (int j = 0; j < in_size.second; j++) {
                ret[i][j] = in[0][i * in_size.second + j];
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

    void saveCheckpoint(ofstream& ofstream)
    {
    }

    void loadCheckpoint(ifstream& ifstream)
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
    virtual Mat& forward(Mat& in) = 0;
    virtual Mat backward(Mat& in) = 0;
    virtual void randomize(default_random_engine& e) = 0;
    virtual void learn(Optimizer* optimizer) = 0;
    virtual void saveCheckpoint(ofstream& ofstream) = 0;
    virtual void loadCheckpoint(ifstream& ifstream) = 0;
};

class DenseLayer : public LinearLayer {
protected:
    Mat w, b;
    init::Initializer* u;

public:
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;
    DenseLayer(int in, int out, init::Initializer* u)
        : LinearLayer(in, out)
        , w(in, out)
        , b(1, out)
        , delta_w(in, out)
        , delta_b(1, out)
        , nabla_w(in, out)
        , nabla_b(1, out)
        , u(u)
    {
        nabla_w.clear(), nabla_b.clear();
    }

    DenseLayer(int in, int out, init::Type type = init::KAIMING)
        : DenseLayer(in, out, getInit(type, in + out))
    {
    }

    ~DenseLayer()
    {
        delete u;
    }

    Mat& forward(Mat& in)
    {
        y = in * w + b;
        x = in;
        return y;
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
        w.randomize(u, e), b.randomize(u, e);
    }

    void learn(Optimizer* optimizer)
    {
        w = optimizer->optimize(w, nabla_w);
        b = optimizer->optimize(b, nabla_b);
        nabla_w.clear();
        nabla_b.clear();
    }

    void saveCheckpoint(ofstream& ofstream)
    {
        ofstream << w << b;
    }

    void loadCheckpoint(ifstream& ifstream)
    {
        ifstream >> w >> b;
    }
};

class SigmoidLayer : public Layer {
protected:
    Mat x;

public:
    SigmoidLayer()
    {
    }
    Mat& forward(Mat& in)
    {
        x = in;
        mutil::sigmoid(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        mutil::sigmoid_prime(x);
        return in.dot(x);
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }

    void saveCheckpoint(ofstream& ofstream)
    {
    }

    void loadCheckpoint(ifstream& ifstream)
    {
    }
};

class RELULayer : public Layer {
protected:
    Mat x;

public:
    RELULayer()
    {
    }
    Mat& forward(Mat& in)
    {
        x = in;
        mutil::relu(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        mutil::relu_prime(x);
        return in.dot(x);
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }

    void saveCheckpoint(ofstream& ofstream)
    {
    }

    void loadCheckpoint(ifstream& ifstream)
    {
    }
};

class ConvLayer : public Layer {
protected:
    Mat x, y;
    init::Initializer* u;
    vector<int> in_size, kernel_size;
    pair<int, int> out_size;

public:
    int stride, padding;
    Mat w, b;
    Mat delta_w, delta_b;
    Mat nabla_w, nabla_b;

public:
    ConvLayer(int height, int width, int channel, int kernel_height, int kernel_width, int kernel_count, int stride, int padding, init::Initializer* u)
        : w(channel * kernel_count, kernel_height * kernel_width)
        , b(kernel_count, 1)
        , delta_w(channel * kernel_count, kernel_height * kernel_width)
        , delta_b(kernel_count, 1)
        , nabla_w(channel * kernel_count, kernel_height * kernel_width)
        , nabla_b(kernel_count, 1)
        , u(u)
    {
        this->in_size = { channel, height, width };
        this->kernel_size = { kernel_count, kernel_height, kernel_width };
        this->stride = stride;
        this->padding = padding;
        nabla_w.clear(), nabla_b.clear();
        out_size = mutil::compute_output_size(in_size[1], in_size[2], kernel_size[1], kernel_size[2], stride, padding);
        y = Mat(kernel_size[0], out_size.first * out_size.second);
    }

    ConvLayer(int height, int width, int channel, int kernel_height, int kernel_width, int kernel_count, int stride, int padding, init::Type type = init::KAIMING, bool forward = true)
        : ConvLayer(height, width, channel, kernel_height, kernel_width, kernel_count, stride, padding, getInit(type, forward ? channel * kernel_height * kernel_width : kernel_count * kernel_height * kernel_width))
    {
    }

    Mat& forward(Mat& in)
    {
        Tensor tensor(in_size, in);
        y.clear();
        auto f = [&](int i) {
            Kernel img(in_size[1], in_size[2], tensor[i]);
            for (int j = 0; j < kernel_size[0]; j++) {
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(out_size.first, out_size.second, y[j]);
                mutil::conv(img, kernel, out, stride, padding);
            }
            std::unique_lock<mutex> lock(tasks_mutex);
            --threadCnt;
            lock.unlock();
        };
        for (int i = 0; i < in_size[0]; i++) {
            while (threadCnt > thread::hardware_concurrency())
                ;
            std::unique_lock<mutex> lock(tasks_mutex);
            ++threadCnt;
            thread t(f, i);
            t.detach();
            lock.unlock();
        }
        while (threadCnt)
            ;
        for (int j = 0; j < kernel_size[0]; j++) {
            Kernel out(out_size.first, out_size.second, y[j]);
            out += b[0][j];
        }
        x = in;
        return y;
    }
    Mat backward(Mat& in)
    {
        Tensor img_tensor(in_size, x);
        Tensor delta_tensor({ kernel_size[0], out_size.first, out_size.second }, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        delta_w.clear();
        delta_b.clear();
        auto f1 = [](Kernel img, Kernel kernel, Kernel out, int stride, int padding) {
            mutil::conv(img, kernel, out, stride, padding);
            std::unique_lock<mutex> lock(tasks_mutex);
            --threadCnt;
            lock.unlock();
        };
        auto f2 = [](Kernel img, Kernel kernel, Kernel out, int stride, int padding) {
            mutil::conv_transpose(img, kernel, out, stride, padding);
            std::unique_lock<mutex> lock(tasks_mutex);
            --threadCnt;
            lock.unlock();
        };

        for (int i = 0; i < in_size[0]; i++) {
            Kernel img(in_size[1], in_size[2], img_tensor[i]);
            for (int j = 0; j < kernel_size[0]; j++) {
                Kernel dw_kernel(kernel_size[1], kernel_size[2], delta_w[i * kernel_size[0] + j]);
                Kernel in_kernel(out_size.first, out_size.second, delta_tensor[j]);
                while (threadCnt > thread::hardware_concurrency())
                    ;
                std::unique_lock<mutex> lock(tasks_mutex);
                ++threadCnt;
                thread t(f1, img, in_kernel, dw_kernel, stride, padding);
                t.detach();
                lock.unlock();
                delta_b[j][0] += mutil::sum(in_kernel);
                Kernel kernel(kernel_size[1], kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                while (threadCnt > thread::hardware_concurrency())
                    ;
                std::unique_lock<mutex> lock2(tasks_mutex);
                ++threadCnt;
                thread t2(f2, in_kernel, kernel, out, stride, padding);
                t2.detach();
                lock2.unlock();
            }
        }
        while (threadCnt)
            ;
        nabla_w += delta_w;
        nabla_b += delta_b;
        return ret;
    }

    void randomize(default_random_engine& e)
    {
        w.randomize(u, e), b.randomize(u, e);
    }

    void learn(Optimizer* optimizer)
    {
        w = optimizer->optimize(w, nabla_w);
        b = optimizer->optimize(b, nabla_b);
        nabla_w.clear();
        nabla_b.clear();
    }

    void saveCheckpoint(ofstream& ofstream)
    {
        ofstream << w << b;
    }

    void loadCheckpoint(ifstream& ifstream)
    {
        ifstream >> w >> b;
    }
};

class PoolingLayer : public Layer {
    enum Type {
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
    PoolingLayer(int height, int width, int channel, pair<int, int> size, int stride, Type type = MAX)
        : x(channel, height * width)
        , y(channel, height * width)
        , pool_size(size)
        , stride(stride)
        , type(type)
    {
        in_size = { channel, height, width };
        out_size = mutil::compute_output_size(in_size[1], in_size[2], size.first, size.second, stride, 0);
        y = Mat(channel, out_size.first * out_size.second);
    }

    Mat& forward(Mat& in)
    {
        Tensor tensor(in_size, in);
        y.clear();
        if (type == MAX) {
            for (int i = 0; i < in_size[0]; i++) {
                Kernel img(in_size[1], in_size[2], tensor[i]);
                Kernel out(out_size.first, out_size.second, y[i]);
                mutil::max_pooling(img, out, pool_size, stride);
            }
        }
        if (type == MEAN) {
            for (int i = 0; i < in_size[0]; i++) {
                Kernel img(in_size[1], in_size[2], tensor[i]);
                Kernel out(out_size.first, out_size.second, y[i]);
                mutil::mean_pooling(img, out, pool_size, stride);
            }
        }
        x = in;
        return y;
    }

    Mat backward(Mat& in)
    {
        Tensor img_tensor(in_size, x);
        Tensor delta_tensor({ in_size[0], out_size.first, out_size.second }, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        if (type == MAX) {
            for (int i = 0; i < in_size[0]; i++) {
                Kernel img(in_size[1], in_size[2], img_tensor[i]);
                Kernel delta(out_size.first, out_size.second, delta_tensor[i]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::max_pooling_prime(img, delta, out, pool_size, stride);
            }
        }
        if (type == MEAN) {
            for (int i = 0; i < in_size[0]; i++) {
                Kernel delta(out_size.first, out_size.second, delta_tensor[i]);
                Kernel out(in_size[1], in_size[2], ret[i]);
                mutil::mean_pooling_prime(delta, out, pool_size, stride);
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

    void saveCheckpoint(ofstream& ofstream)
    {
    }

    void loadCheckpoint(ifstream& ifstream)
    {
    }
};

class SoftmaxLayer : public Layer {
protected:
public:
    SoftmaxLayer()
    {
    }
    Mat& forward(Mat& in)
    {
        mutil::softmax(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        return in;
    }

    void randomize(default_random_engine& e)
    {
    }

    void learn(Optimizer* optimizer)
    {
    }

    void saveCheckpoint(ofstream& ofstream)
    {
    }

    void loadCheckpoint(ifstream& ifstream)
    {
    }
};

#endif