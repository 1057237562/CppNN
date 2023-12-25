#ifndef INITIALIZER_CPP
#define INITIALIZER_CPP

#include <random>

namespace init {
enum Type {
    UNIFORM,
    NORMAL,
    XAVIER,
    KAIMING
};

class Initializer {
public:
    virtual float generate(std::default_random_engine& e) = 0;

    float operator()(std::default_random_engine& e)
    {
        return generate(e);
    }

    virtual ~Initializer() { }
};

class UniformInit : public Initializer {
    float a, b;

public:
    UniformInit(float a = 0, float b = 1)
        : a(a)
        , b(b)
    {
    }
    float generate(std::default_random_engine& e)
    {
        std::uniform_real_distribution<float> u(a, b);
        return u(e);
    }
};

class NormalInit : public Initializer {
    float a, b;

public:
    NormalInit(float a = 0, float b = 1)
        : a(a)
        , b(b)
    {
    }
    float generate(std::default_random_engine& e)
    {
        std::normal_distribution<float> u(a, b);
        return u(e);
    }
};

class XavierInit : public Initializer {
    enum Method {
        UNIFORM,
        NORMAL
    };
    int n;
    Method mtd;

public:
    XavierInit(int n, Method mtd = NORMAL)
        : n(n)
        , mtd(mtd)
    {
    }

    float generate(std::default_random_engine& e)
    {
        if (mtd == NORMAL) {
            std::normal_distribution<float> u(0, sqrt(2.0f / n));
            return u(e);
        }
        if (mtd == UNIFORM) {
            std::uniform_real_distribution<float> u(-sqrt(6.0f / n), sqrt(6.0f / n));
            return u(e);
        }
        return 0;
    }
};

class KaimingInit : public Initializer {
    enum Method {
        UNIFORM,
        NORMAL
    };
    float bound;
    Method mtd;

public:
    KaimingInit(int n, int a2 = 5, Method mtd = NORMAL)
        : mtd(mtd)
    {
        bound = (1 + a2) * n;
    }

    float generate(std::default_random_engine& e)
    {
        if (mtd == NORMAL) {
            std::normal_distribution<float> u(0, sqrt(2.0f / bound));
            return u(e);
        }
        if (mtd == UNIFORM) {
            std::uniform_real_distribution<float> u(-sqrt(6.0f / bound), sqrt(6.0f / bound));
            return u(e);
        }
        return 0;
    }
};
}

#endif