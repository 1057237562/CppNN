#ifndef MUTIL_CPP
#define MUTIL_CPP

#include <assert.h>
#include <cmath>
#include <random>
#include <time.h>
#include <utility>
#include <vector>

using namespace std;

namespace mutil
{
    enum RandomMethod
    {
        NORMAL,
        UNIFORM
    };
    static int copyCount = 0;
    static int constructTime = 0;
    static int multiplyTime = 0;
    static int multiplyCount = 0;

    class Vec
    {

    public:
        vector<float> val;
        int length;
        Vec(int len)
        {
            length = len;
            val = vector<float>(length);
        }

        float &operator[](int index)
        {
            assert(index >= 0 && index < length);
            return val[index];
        }

        void randomize(default_random_engine &e, RandomMethod method = NORMAL)
        {
            if (method == UNIFORM)
            {
                uniform_real_distribution<float> u(0, 1);
                for (int i = 0; i < length; i++)
                {
                    val[i] = u(e);
                }
            }
            if(method == NORMAL)
            {
                normal_distribution<float> u(0, 1);
                for (int i = 0; i < length; i++)
                {
                    val[i] = u(e);
                }
            }
        }
    };

    class Mat
    {
        vector<float> val;

    public:
        pair<int, int> size;
        Mat() {}
        Mat(int m, int n)
            : val(m * n)
        {
            size = {m, n};
        }

        Mat(int m,int n,vector<float> &v)
            : val(v)
        {
            size = {m, n};
        }

        auto operator[](int index)
        {
            assert(index >= 0 && index < size.first);
            return val.begin() + index * size.second;
        }

        Mat &dot(Mat &other)
        {
            assert(size.first == other.size.first && size.second == other.size.second);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] *= other[i][j];
                }
            }
            return *this;
        }

        Mat &operator+(Mat &other)
        {
            assert(size.first == other.size.first && size.second == other.size.second);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] += other[i][j];
                }
            }
            return *this;
        }

        Mat &operator+=(Mat &other)
        {
            assert(size.first == other.size.first && size.second == other.size.second);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] += other[i][j];
                }
            }
            return *this;
        }

        Mat &operator-(Mat &other)
        {
            assert(size.first == other.size.first && size.second == other.size.second);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] -= other[i][j];
                }
            }
            return *this;
        }

        Mat &operator-=(Mat &other)
        {
            assert(size.first == other.size.first && size.second == other.size.second);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] -= other[i][j];
                }
            }
            return *this;
        }

        Mat operator*(Mat &other)
        {
            assert(size.second == other.size.first);
            auto start = clock();
            Mat res(size.first, other.size.second);
            float r;
            for (int i = 0; i < size.first; ++i)
                for (int k = 0; k < size.second; ++k)
                {
                    r = (*this)[i][k];
                    for (int j = 0; j < other.size.second; ++j)
                        res[i][j] += other[k][j] * r;
                }
            auto end = clock();
            multiplyTime += end - start;
            ++multiplyCount;
            return res;
        }

        Mat &operator*(float &theta)
        {
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] *= theta;
                }
            }
            return *this;
        }

        Mat transpose()
        {
            Mat res(size.second, size.first);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    res[j][i] = (*this)[i][j];
                }
            }
            return res;
        }

        void randomize(default_random_engine &e)
        {
            normal_distribution<float> u(0, 1);
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] = u(e);
                }
            }
        }

        void clear()
        {
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    (*this)[i][j] = 0;
                }
            }
        }
    };

    class Kernel {
        vector<float>::iterator val;
        public:
            pair<int, int> size;
            Kernel(int m,int n,vector<float>::iterator val) : val(val) {
                size = {m, n};
            }

            auto operator[](int index)
            {
                assert(index >= 0 && index < size.first);
                return val + index * size.second;
            }

            void operator+=(Kernel& other)
            {
                assert(size.first == other.size.first && size.second == other.size.second);
                for (int i = 0; i < size.first; i++)
                {
                    for (int j = 0; j < size.second; j++)
                    {
                        (*this)[i][j] += other[i][j];
                    }
                }
            }

            void operator+=(float& theta)
            {
                for (int i = 0; i < size.first; i++)
                {
                    for (int j = 0; j < size.second; j++)
                    {
                        (*this)[i][j] += theta;
                    }
                }
            }

    };

    class Tensor {
        vector<int> dimension;
        vector<float>::iterator val;
        int size = 1;
    public:
        Tensor(vector<int> dimension,vector<float>::iterator val) : dimension(dimension),val(val) {
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
            dimension.erase(dimension.begin());
            return val + index * (size / dim0);
        }
    };

    Mat &sigmoid(Mat &in)
    {
        for (int i = 0; i < in.size.first; i++)
        {
            for (int j = 0; j < in.size.second; j++)
            {
                in[i][j] = 1 / (1 + exp(-in[i][j]));
            }
        }
        return in;
    }

    Mat &sigmoid_prime(Mat &in)
    {
        for (int i = 0; i < in.size.first; i++)
        {
            for (int j = 0; j < in.size.second; j++)
            {
                float s = 1 / (1 + exp(-in[i][j]));
                in[i][j] = s * (1 - s);
            }
        }
        return in;
    }

    Mat &relu(Mat &in)
    {
        for (int i = 0; i < in.size.first; i++)
        {
            for (int j = 0; j < in.size.second; j++)
            {
                in[i][j] = max(0.0f, in[i][j]);
            }
        }
        return in;
    }

    Mat &relu_prime(Mat &in)
    {
        for (int i = 0; i < in.size.first; i++)
        {
            for (int j = 0; j < in.size.second; j++)
            {
                in[i][j] = in[i][j] > 0 ? 1 : 0;
            }
        }
        return in;
    }

    pair<int,int> compute_output_size(int in_height, int in_width, int kernel_height, int kernel_width, int stride, int padding)
    {
        int out_height = (in_height - kernel_height + 2 * padding) / stride + 1;
        int out_width = (in_width - kernel_width + 2 * padding) / stride + 1;
        return {out_height,out_width};
    }

    void conv(Kernel &in, Kernel &kernel,Kernel& out, int stride, int padding)
    {
        for (int i = 0; i < out.size.first; i++)
        {
            for (int j = 0; j < out.size.second; j++)
            {
                float sum = 0;
                for (int k = 0; k < kernel.size.first; k++)
                {
                    for (int l = 0; l < kernel.size.second; l++)
                    {
                        int x = i * stride + k - padding;
                        int y = j * stride + l - padding;
                        if (x >= 0 && x < in.size.first && y >= 0 && y < in.size.second)
                        {
                            sum += in[x][y] * kernel[k][l];
                        }
                    }
                }
                out[i][j] += sum;
            }
        }
    }
}

#endif