#ifndef MUTIL_CPP
#define MUTIL_CPP

#include "initializer.cpp"
#include <assert.h>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <time.h>
#include <utility>
#include <vector>

using namespace std;

namespace mutil {
static int copyCount = 0;
static int constructTime = 0;
static int multiplyTime = 0;
static int multiplyCount = 0;

class Vec {

public:
    vector<float> val;
    int length;
    Vec(int len)
    {
        length = len;
        val = vector<float>(length);
    }

    float& operator[](int index)
    {
        assert(index >= 0 && index < length);
        return val[index];
    }

    void randomize(init::Initializer* init, default_random_engine& e)
    {
        for (int i = 0; i < length; i++) {
            val[i] = init->generate(e);
        }
    }
};

class Mat {
    vector<float> val;

public:
    pair<int, int> size;
    Mat() { }
    Mat(int m, int n)
        : val(m * n)
    {
        size = { m, n };
        ++constructTime;
    }

    Mat(int m, int n, vector<float>& v)
        : val(v)
    {
        size = { m, n };
        ++constructTime;
    }

    auto operator[](int index)
    {
        assert(index >= 0 && index < size.first);
        return val.begin() + index * size.second;
    }

    Mat& dot(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] *= other[i][j];
            }
        }
        return *this;
    }

    Mat& operator+(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] += other[i][j];
            }
        }
        return *this;
    }

    Mat& operator+=(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] += other[i][j];
            }
        }
        return *this;
    }

    Mat& operator-(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    Mat& operator-=(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    Mat operator*(Mat& other)
    {
        assert(size.second == other.size.first);
        auto start = clock();
        Mat res(size.first, other.size.second);
        float r;
        for (int i = 0; i < size.first; ++i)
            for (int k = 0; k < size.second; ++k) {
                r = (*this)[i][k];
                for (int j = 0; j < other.size.second; ++j)
                    res[i][j] += other[k][j] * r;
            }
        auto end = clock();
        multiplyTime += end - start;
        ++multiplyCount;
        return res;
    }

    Mat& operator*(float& theta)
    {
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] *= theta;
            }
        }
        return *this;
    }

    Mat transpose()
    {
        Mat res(size.second, size.first);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                res[j][i] = (*this)[i][j];
            }
        }
        return res;
    }

    void randomize(init::Initializer* init, default_random_engine& e)
    {
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] = init->generate(e);
            }
        }
    }

    void clear()
    {
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] = 0;
            }
        }
    }
    friend ostream& operator<<(ostream& os, const Mat& mat);
    friend ofstream& operator<<(ofstream& os, const Mat& mat);
    friend istream& operator>>(istream& is, Mat& mat);
    friend ifstream& operator>>(ifstream& is, Mat& mat);
};

ostream& operator<<(ostream& os, const Mat& mat)
{
    os << mat.size.first << ' ' << mat.size.second << ' ';
    for (int i = 0; i < mat.val.size(); i++) {
        os << mat.val[i] << ' ';
    }
    return os;
}

ofstream& operator<<(ofstream& os, const Mat& mat)
{
    os << mat.size.first << ' ' << mat.size.second << ' ';
    for (int i = 0; i < mat.val.size(); i++) {
        os << mat.val[i] << ' ';
    }
    return os;
}

istream& operator>>(istream& is, Mat& mat)
{
    is >> mat.size.first >> mat.size.second;
    mat.val = vector<float>(mat.size.first * mat.size.second);
    for (int i = 0; i < mat.val.size(); i++) {
        is >> mat.val[i];
    }
    return is;
}

ifstream& operator>>(ifstream& is, Mat& mat)
{
    is >> mat.size.first >> mat.size.second;
    mat.val = vector<float>(mat.size.first * mat.size.second);
    for (int i = 0; i < mat.val.size(); i++) {
        is >> mat.val[i];
    }
    return is;
}

class Kernel {
    vector<float>::iterator val;

public:
    pair<int, int> size;
    Kernel(int m, int n, vector<float>::iterator val)
        : val(val)
    {
        size = { m, n };
    }

    auto operator[](int index)
    {
        assert(index >= 0 && index < size.first);
        return val + index * size.second;
    }

    void operator+=(Kernel& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] += other[i][j];
            }
        }
    }

    void operator+=(float& theta)
    {
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                (*this)[i][j] += theta;
            }
        }
    }

    Mat to_Mat()
    {
        Mat res(size.first, size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                res[i][j] = (*this)[i][j];
            }
        }
        return res;
    }
};

class Tensor {
    vector<int> dimension;
    vector<float>::iterator val;
    int size = 1;

public:
    Tensor(vector<int> dimension, vector<float>::iterator val)
        : dimension(dimension)
        , val(val)
    {
        for (int i = 0; i < dimension.size(); i++) {
            size *= dimension[i];
        }
    }

    Tensor(vector<int> dimension, Mat& data)
    {
        for (int i = 0; i < dimension.size(); i++) {
            size *= dimension[i];
        }
        this->dimension = dimension;
        this->val = data[0];
    }

    auto operator[](int index)
    {
        assert(index >= 0 && index < dimension[0]);
        int dim0 = dimension[0];
        return val + index * (size / dim0);
    }
};

void sigmoid(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            in[i][j] = 1 / (1 + exp(-in[i][j]));
        }
    }
}

Mat& sigmoid_prime(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            float s = 1 / (1 + exp(-in[i][j]));
            in[i][j] = s * (1 - s);
        }
    }
    return in;
}

void relu(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            in[i][j] = max(0.0f, in[i][j]);
        }
    }
}

void relu_prime(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            in[i][j] = in[i][j] > 0 ? 1 : 0;
        }
    }
}

pair<int, int> compute_output_size(int in_height, int in_width, int kernel_height, int kernel_width, int stride, int padding)
{
    int out_height = (in_height - kernel_height + 2 * padding) / stride + 1;
    int out_width = (in_width - kernel_width + 2 * padding) / stride + 1;
    return { out_height, out_width };
}

void conv(Kernel& in, Kernel& kernel, Kernel& out, int stride, int padding)
{
    for (int i = 0; i < out.size.first; i++) {
        for (int j = 0; j < out.size.second; j++) {
            for (int k = 0; k < kernel.size.first; k++) {
                for (int l = 0; l < kernel.size.second; l++) {
                    int x = i * stride + k - padding;
                    int y = j * stride + l - padding;
                    if (x >= 0 && x < in.size.first && y >= 0 && y < in.size.second) {
                        out[i][j] += in[x][y] * kernel[k][l];
                    }
                }
            }
        }
    }
}

void conv_transpose(Kernel& in, Kernel& kernel, Kernel& out, int stride, int padding)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            for (int k = 0; k < kernel.size.first; k++) {
                for (int l = 0; l < kernel.size.second; l++) {
                    int x = i * stride + k - padding;
                    int y = j * stride + l - padding;
                    if (x >= 0 && x < out.size.first && y >= 0 && y < out.size.second) {
                        out[x][y] += in[i][j] * kernel[k][l];
                    }
                }
            }
        }
    }
}

float sum(Kernel& in)
{
    float ret = 0;
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            ret += in[i][j];
        }
    }
    return ret;
}

void max_pooling(Kernel& in, Kernel& out, pair<int, int>& size, int stride)
{
    for (int i = 0; i < out.size.first; i++) {
        for (int j = 0; j < out.size.second; j++) {
            float max = FLT_MIN;
            for (int k = 0; k < size.first; k++) {
                for (int l = 0; l < size.second; l++) {
                    int x = i * stride + k;
                    int y = j * stride + l;
                    if (x >= 0 && x < in.size.first && y >= 0 && y < in.size.second) {
                        if (in[x][y] > max) {
                            max = in[x][y];
                        }
                    }
                }
            }
            out[i][j] = max;
        }
    }
}

void mean_pooling(Kernel& in, Kernel& out, pair<int, int>& size, int stride)
{
    for (int i = 0; i < out.size.first; i++) {
        for (int j = 0; j < out.size.second; j++) {
            float sum = 0;
            for (int k = 0; k < size.first; k++) {
                for (int l = 0; l < size.second; l++) {
                    int x = i * stride + k;
                    int y = j * stride + l;
                    if (x >= 0 && x < in.size.first && y >= 0 && y < in.size.second) {
                        sum += in[x][y];
                    }
                }
            }
            out[i][j] = sum / (size.first * size.second);
        }
    }
}

void max_pooling_prime(Kernel& img, Kernel& delta, Kernel& out, pair<int, int>& size, int stride)
{
    for (int i = 0; i < delta.size.first; i++) {
        for (int j = 0; j < delta.size.second; j++) {
            float max = FLT_MIN;
            int max_x = 0;
            int max_y = 0;
            for (int k = 0; k < size.first; k++) {
                for (int l = 0; l < size.second; l++) {
                    int x = i * stride + k;
                    int y = j * stride + l;
                    if (x >= 0 && x < img.size.first && y >= 0 && y < img.size.second) {
                        if (img[x][y] > max) {
                            max = img[x][y];
                            max_x = x;
                            max_y = y;
                        }
                    }
                }
            }
            out[max_x][max_y] = delta[i][j];
        }
    }
}

void mean_pooling_prime(Kernel& delta, Kernel& out, pair<int, int>& size, int stride)
{
    for (int i = 0; i < delta.size.first; i++) {
        for (int j = 0; j < delta.size.second; j++) {
            for (int k = 0; k < size.first; k++) {
                for (int l = 0; l < size.second; l++) {
                    int x = i * stride + k;
                    int y = j * stride + l;
                    if (x >= 0 && x < out.size.first && y >= 0 && y < out.size.second) {
                        out[x][y] = delta[i][j] / (size.first * size.second);
                    }
                }
            }
        }
    }
}

void softmax(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        float sum = 0;
        float max = FLT_MIN;
        for (int j = 0; j < in.size.second; j++) {
            if (in[i][j] > max) {
                max = in[i][j];
            }
        }
        for (int j = 0; j < in.size.second; j++) {
            sum += exp(in[i][j] - max);
        }
        for (int j = 0; j < in.size.second; j++) {
            in[i][j] = exp(in[i][j] - max) / sum;
        }
    }
}

inline float im2col_get_pixel(Mat& in, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;
    return in[channel][col + width * row];
}

void im2col(Mat& in, int channels, int height, int width, pair<int, int> ksize, int stride, int pad, Mat& out)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize.first) / stride + 1;
    int width_col = (width + 2 * pad - ksize.second) / stride + 1;

    int channels_col = channels * ksize.first * ksize.second;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize.second;
        int h_offset = (c / ksize.second) % ksize.first;
        int c_im = c / ksize.first / ksize.second;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                assert(col_index < out.size.first * out.size.second);
                out[0][col_index] = im2col_get_pixel(in, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(Mat& im, int height, int width, int channels, int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return;
    im[channel][col + width * row] += val;
}

void col2im(Mat& in, int channels, int height, int width, pair<int, int> ksize, int stride, int pad, Mat& out)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize.first) / stride + 1;
    int width_col = (width + 2 * pad - ksize.second) / stride + 1;

    int channels_col = channels * ksize.first * ksize.second;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize.second;
        int h_offset = (c / ksize.second) % ksize.first;
        int c_im = c / ksize.first / ksize.second;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                assert(col_index < in.size.first * in.size.second);
                float val = in[0][col_index];
                col2im_add_pixel(out, height, width, channels, im_row, im_col, c_im, pad, val);
            }
        }
    }
}

void multiply(Kernel& a, Kernel& b, Kernel& res)
{
    assert(a.size.second == b.size.first);
    assert(res.size.first == a.size.first && res.size.second == b.size.second);
    auto start = clock();
    float r;
    for (int i = 0; i < a.size.first; ++i)
        for (int k = 0; k < a.size.second; ++k) {
            r = a[i][k];
            for (int j = 0; j < b.size.second; ++j)
                res[i][j] += b[k][j] * r;
        }
    auto end = clock();
    multiplyTime += end - start;
    ++multiplyCount;
}
}

#endif