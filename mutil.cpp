#ifndef MUTIL_CPP
#define MUTIL_CPP

#include <assert.h>
#include <cmath>
#include <random>
#include <time.h>
#include <utility>
#include <vector>

using namespace std;

namespace mutil {

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

    void randomize(default_random_engine& e)
    {
        normal_distribution<float> u(0, 1);
        for (int i = 0; i < length; i++) {
            val[i] = u(e);
        }
    }
};

class Mat {
public:
    vector<Vec> val;
    pair<int, int> size;
    Mat(int m, int n)
    {
        val = vector<Vec>(m, Vec(n));
        size = { m, n };
    }

    Mat(vector<Vec> v)
    {
        val = v;
        size = { (int)v.size(), (int)v[0].length };
    }

    Vec& operator[](int index)
    {
        assert(index >= 0 && index < size.first);
        return val[index];
    }

    Mat& dot(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                val[i][j] *= other[i][j];
            }
        }
        return *this;
    }

    Mat& operator+(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                val[i][j] += other[i][j];
            }
        }
        return *this;
    }

    Mat& operator+=(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                val[i][j] += other[i][j];
            }
        }
        return *this;
    }

    Mat& operator-(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                val[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    Mat& operator-=(Mat& other)
    {
        assert(size.first == other.size.first && size.second == other.size.second);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                val[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    Mat operator*(Mat& other)
    {
        assert(size.second == other.size.first);
        auto start = clock();
        Mat res(size.first, other.size.second);
        float Bcolj[size.second]; // Accelerate matrix multiplication
        for (int j = 0; j < other.size.second; j++) {
            for (int k = 0; k < size.second; k++) {
                Bcolj[k] = other[k][j];
            }
            for (int i = 0; i < size.first; i++) {
                Vec& Arowi = val[i];
                float s = 0;
                for (int k = 0; k < size.second; k++) {
                    s += Arowi[k] * Bcolj[k];
                }
                res[i][j] = s;
            }
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
                val[i][j] *= theta;
            }
        }
        return *this;
    }

    Mat transpose()
    {
        Mat res(size.second, size.first);
        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                res[j][i] = val[i][j];
            }
        }
        return res;
    }

    void randomize(default_random_engine& e)
    {
        for (auto& vec : val) {
            vec.randomize(e);
        }
    }

    void clear()
    {
        for (auto& vec : val) {
            for (int i = 0; i < vec.length; i++) {
                vec[i] = 0;
            }
        }
    }
};

Mat& sigmoid(Mat& in)
{
    for (int i = 0; i < in.size.first; i++) {
        for (int j = 0; j < in.size.second; j++) {
            in[i][j] = 1 / (1 + exp(-in[i][j]));
        }
    }
    return in;
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
}

#endif