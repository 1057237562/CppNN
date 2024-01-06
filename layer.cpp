#ifndef LAYER_CPP
#define LAYER_CPP

#include "debug.cpp"
#include "mutil.cpp"
#include "optimizer.cpp"
#include <fstream>

using namespace mutil;

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

class ActivationLayer : public Layer {
protected:
    Mat x;

public:
    ActivationLayer()
    {
    }
    virtual Mat& forward(Mat& in) = 0;
    virtual Mat backward(Mat& in) = 0;
    void randomize(default_random_engine& e) {};
    void learn(Optimizer* optimizer) {};
    void saveCheckpoint(ofstream& ofstream) {};
    void loadCheckpoint(ifstream& ifstream) {};
};

class SigmoidLayer : public ActivationLayer {
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
};

class RELULayer : public ActivationLayer {
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
};

class TanhLayer : public ActivationLayer {
public:
    TanhLayer()
    {
    }
    Mat& forward(Mat& in)
    {
        x = in;
        mutil::tanh(in);
        return in;
    }
    Mat backward(Mat& in)
    {
        mutil::tanh_prime(x);
        return in.dot(x);
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
        return in * w.transpose();
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
        y.clear();
        Mat data_col(in_size[0], out_size.first * out_size.second * kernel_size[1] * kernel_size[2]);
        mutil::im2col(in, in_size[0], in_size[1], in_size[2], { kernel_size[1], kernel_size[2] }, stride, padding, data_col);
        for (int i = 0; i < in_size[0]; i++) {
            Kernel col(kernel_size[1] * kernel_size[2], out_size.first * out_size.second, data_col[i]);
            for (int j = 0; j < kernel_size[0]; j++) {
                Kernel kernel(1, kernel_size[1] * kernel_size[2], w[i * kernel_size[0] + j]);
                Kernel out(1, out_size.first * out_size.second, y[j]);
                mutil::multiply(kernel, col, out);
            }
        }
        for (int j = 0; j < kernel_size[0]; j++) {
            Kernel out(out_size.first, out_size.second, y[j]);
            out += b[0][j];
        }
        x = in;
        return y;
    }
    Mat backward(Mat& in)
    {
        Mat data_col(in_size[0], out_size.first * out_size.second * kernel_size[1] * kernel_size[2]);
        mutil::im2col(x, in_size[0], in_size[1], in_size[2], out_size, stride, padding, data_col);
        Mat ret_img(in_size[0], out_size.first * out_size.second * kernel_size[1] * kernel_size[2]);
        Tensor delta_tensor({ kernel_size[0], out_size.first, out_size.second }, in);
        Mat ret(in_size[0], in_size[1] * in_size[2]);
        ret.clear();
        delta_w.clear();
        delta_b.clear();
        for (int i = 0; i < in_size[0]; i++) {
            Kernel img(out_size.first * out_size.second, kernel_size[1] * kernel_size[2], data_col[i]);
            for (int j = 0; j < kernel_size[0]; j++) {
                Kernel dw_kernel(1, kernel_size[1] * kernel_size[2], delta_w[i * kernel_size[0] + j]);
                Kernel in_kernel(1, out_size.first * out_size.second, delta_tensor[j]);
                mutil::multiply(in_kernel, img, dw_kernel);
                delta_b[j][0] += mutil::sum(in_kernel);
                Kernel kernel(kernel_size[1] * kernel_size[2], 1, w[i * kernel_size[0] + j]);
                Kernel out(kernel_size[1] * kernel_size[2], out_size.first * out_size.second, ret_img[i]);
                mutil::multiply(kernel, in_kernel, out);
            }
        }
        mutil::col2im(ret_img, in_size[0], in_size[1], in_size[2], { kernel_size[1], kernel_size[2] }, stride, padding, ret);
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

class RNNLayer : public LinearLayer {
protected:
    Mat h, h0;
    init::Initializer* u;
    ActivationLayer* ac;
    int hidden_size;

public:
    Mat wi, wh, b;
    Mat delta_wi, delta_wh, delta_b;
    Mat nabla_wi, nabla_wh, nabla_b;
    RNNLayer(int in_size, int hidden_size, init::Initializer* u, ActivationLayer* activationLayer = new TanhLayer())
        : LinearLayer(in_size, hidden_size)
        , hidden_size(hidden_size)
        , h(1, hidden_size)
        , h0(1, hidden_size)
        , wi(in_size, hidden_size)
        , wh(hidden_size, hidden_size)
        , b(1, hidden_size)
        , delta_wi(in_size, hidden_size)
        , delta_wh(hidden_size, hidden_size)
        , delta_b(1, hidden_size)
        , nabla_wi(in_size, hidden_size)
        , nabla_wh(hidden_size, hidden_size)
        , nabla_b(1, hidden_size)
        , u(u)
        , ac(activationLayer)
    {
    }

    RNNLayer(int in_size, int hidden_size, init::Type type = init::KAIMING, ActivationLayer* activationLayer = new TanhLayer()) // unsafe
        : RNNLayer(in_size, hidden_size, getInit(type, hidden_size), activationLayer)
    {
    }

    Mat& forward(Mat& in)
    {
        h = y;
        y = (h * wh) + (in * wi) + b; // concat or plus
        ac->forward(y);
        return y;
    }
    Mat backward(Mat& in)
    {
        Mat delta_h_prime = ac->backward(in);
        delta_wi = x.transpose() * delta_h_prime;
        delta_wh = h0.transpose() * delta_h_prime;
        delta_b = delta_h_prime;
        nabla_wi += delta_wi;
        nabla_wh += delta_wh;
        nabla_b += delta_b;
        return delta_h_prime * wi.transpose();
    }
    void randomize(default_random_engine& e)
    {
        wi.randomize(u, e), wh.randomize(u, e), b.randomize(u, e);
    }
    void learn(Optimizer* optimizer)
    {
        wi = optimizer->optimize(wi, nabla_wi);
        wh = optimizer->optimize(wh, nabla_wh);
        b = optimizer->optimize(b, nabla_b);
        nabla_wi.clear();
        nabla_wh.clear();
        nabla_b.clear();
    }
    void saveCheckpoint(ofstream& ofstream)
    {
        ofstream << wi << wh << b;
    }
    void loadCheckpoint(ifstream& ifstream)
    {
        ifstream >> wi >> wh >> b;
    }
};

class LSTMLayer : public LinearLayer {
protected:
    Mat h, c, c0;
    Mat f, i, ct, o;
    init::Initializer* u;
    int hidden_size;

public:
    Mat wf, wi, wc, wo, bf, bi, bo, bc;
    Mat delta_wf, delta_wi, delta_wc, delta_wo, delta_bf, delta_bi, delta_bo, delta_bc;
    Mat nabla_wf, nabla_wi, nabla_wc, nabla_wo, nabla_bf, nabla_bi, nabla_bo, nabla_bc;
    LSTMLayer(int in_size, int hidden_size, init::Initializer* u)
        : LinearLayer(in_size, hidden_size)
        , hidden_size(hidden_size)
        , h(1, hidden_size)
        , c(1, hidden_size)
        , c0(1, hidden_size)
        , wf(in_size + hidden_size, hidden_size)
        , wi(in_size + hidden_size, hidden_size)
        , wc(in_size + hidden_size, hidden_size)
        , wo(in_size + hidden_size, hidden_size)
        , bf(1, hidden_size)
        , bi(1, hidden_size)
        , bc(1, hidden_size)
        , bo(1, hidden_size)
        , delta_wf(in_size + hidden_size, hidden_size)
        , delta_wi(in_size + hidden_size, hidden_size)
        , delta_wc(in_size + hidden_size, hidden_size)
        , delta_wo(in_size + hidden_size, hidden_size)
        , delta_bf(1, hidden_size)
        , delta_bi(1, hidden_size)
        , delta_bc(1, hidden_size)
        , delta_bo(1, hidden_size)
        , nabla_wf(in_size + hidden_size, hidden_size)
        , nabla_wi(in_size + hidden_size, hidden_size)
        , nabla_wc(in_size + hidden_size, hidden_size)
        , nabla_wo(in_size + hidden_size, hidden_size)
        , nabla_bf(1, hidden_size)
        , nabla_bi(1, hidden_size)
        , nabla_bc(1, hidden_size)
        , nabla_bo(1, hidden_size)
        , u(u)
    {
    }

    LSTMLayer(int in_size, int hidden_size, init::Type type = init::KAIMING) // unsafe
        : LSTMLayer(in_size, hidden_size, getInit(type, hidden_size))
    {
    }

    Mat& forward(Mat& in)
    {
        x = in;
        h = y;
        c0 = c;
        Mat cat = mutil::concat(h, x);
        f = cat * wf + bf;
        mutil::sigmoid(f);
        i = cat * wi + bf;
        mutil::sigmoid(i);
        ct = cat * wc + bf;
        mutil::tanh(ct);
        o = cat * wo + bf;
        mutil::sigmoid(o);
        c = f.dot(c0) + i.dot(ct);
        y = o.dot(c);
        return y;
    }

    Mat backward(Mat& in)
    {
        Mat delta_h_prime = in;
        Mat delta_c_prime = delta_h_prime * wo.transpose() * mutil::tanh_prime(c);
        Mat delta_f_prime = delta_c_prime.dot(c0) * mutil::sigmoid_prime(c);
        Mat delta_i_prime = delta_c_prime.dot(c) * mutil::sigmoid_prime(c);
        Mat delta_ct_prime = delta_c_prime.dot(i) * mutil::tanh_prime(ct);
        Mat delta_o_prime = delta_h_prime.dot(c) * mutil::sigmoid_prime(o);
        delta_wf = mutil::concat(h, x).transpose() * delta_f_prime;
        delta_wi = mutil::concat(h, x).transpose() * delta_i_prime;
        delta_wc = mutil::concat(h, x).transpose() * delta_ct_prime;
        delta_wo = mutil::concat(h, x).transpose() * delta_o_prime;
        delta_bf = delta_f_prime;
        delta_bi = delta_i_prime;
        delta_bc = delta_ct_prime;
        delta_bo = delta_o_prime;
        nabla_wf += delta_wf;
        nabla_wi += delta_wi;
        nabla_wc += delta_wc;
        nabla_wo += delta_wo;
        nabla_bf += delta_bf;
        nabla_bi += delta_bi;
        nabla_bc += delta_bc;
        nabla_bo += delta_bo;
        return delta_h_prime * wf.transpose() + delta_c_prime * f + delta_i_prime * ct + delta_o_prime * c;
    }
};

#endif