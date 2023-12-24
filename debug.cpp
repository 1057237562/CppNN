#ifndef DEBUG_CPP
#define DEBUG_CPP

#include "mutil.cpp"
#include <iostream>

namespace debug {

void print(mutil::Kernel& k)
{
    for (int i = 0; i < k.size.first; i++) {
        for (int j = 0; j < k.size.second; j++) {
            std::cout << k[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

}

#endif