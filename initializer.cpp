#ifndef INITIALIZER_CPP
#define INITIALIZER_CPP

class Initializer
{
public:
    virtual float generate(default_random_engine &e);
};

#endif