#ifndef MNIST_LOADER

#define MNIST_LOADER

#include "mutil.cpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace mutil;

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
vector<Mat> read_mnist_images(string full_path)
{
    vector<Mat> result;
    ifstream file(full_path + "/t10k-images.idx3-ubyte", ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
            throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        result.resize(number_of_images, Mat(n_rows, n_cols));
        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    result[i][r][c] = int(temp) / 255.0;
                }
            }
        }
    }
    return result;
}

vector<int> read_mnist_labels(string full_path)
{
    vector<int> result;
    ifstream file(full_path + "/t10k-labels.idx1-ubyte", ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        for (int i = 0; i < number_of_labels; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            result.push_back(int(temp));
        }
    }
    return result;
}

#endif