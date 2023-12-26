#include "debug.cpp"
#include "mnist_loader.cpp"
#include "network.cpp"
#include "optimizer.cpp"
#include <fstream>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

#define endl '\n'
using namespace std;
using namespace mutil;

void train()
{
    vector<Mat> train_image = read_mnist_images("./train-images.idx3-ubyte");
    vector<int> train_label = read_mnist_labels("./train-labels.idx1-ubyte");

    vector<Mat> test_image = read_mnist_images("./t10k-images.idx3-ubyte");
    vector<int> test_label = read_mnist_labels("./t10k-labels.idx1-ubyte");

    vector<pair<Mat, Mat>> train_data;
    for (int i = 0; i < train_image.size(); i++) {
        Mat mat(1, 10);
        mat[0][train_label[i]] = 1;
        train_data.push_back({ train_image[i], mat });
    }

    // Network network({ new FlattenLayer(28, 28), new DenseLayer(28 * 28, 16), new SigmoidLayer(), new DenseLayer(16, 16), new SigmoidLayer(), new DenseLayer(16, 10), new SigmoidLayer() }, new SDG(train_data, 0.5, 10));
    Network network({ new ConvLayer(28, 28, 1, 5, 5, 6, 1, 0),
                        new RELULayer(),
                        new PoolingLayer(24, 24, 6, { 2, 2 }, 2),
                        new ConvLayer(12, 12, 6, 5, 5, 16, 1, 0),
                        new RELULayer(),
                        new PoolingLayer(8, 8, 16, { 2, 2 }, 2),
                        new ConvLayer(4, 4, 16, 4, 4, 120, 1, 0),
                        new RELULayer(),
                        new FlattenLayer(),
                        new DenseLayer(120, 84),
                        new RELULayer(),
                        new DenseLayer(84, 10),
                        new SoftmaxLayer() },
        new SDG(train_data, 0.01, 10));
    // Network network({ new ConvLayer(28, 28, 1, 3, 3, 1, 1, 0),
    //                     new PoolingLayer(26, 26, 1, { 2, 2 }, 2),
    //                     new ConvLayer(13, 13, 1, 3, 3, 1, 1, 0),
    //                     new PoolingLayer(11, 11, 1, { 2, 2 }, 2),
    //                     new FlattenLayer(),
    //                     new DenseLayer(25, 16),
    //                     new SigmoidLayer(),
    //                     new DenseLayer(16, 10),
    //                     new SigmoidLayer() },
    //     new SDG(train_data, 0.5, 10));
    // network.init();

    // network.train();

    // ofstream fout("LeNet5.ckpt", ios::out | ios::trunc);

    // network.saveCheckpoint(fout);

    ifstream fin("LeNet5.ckpt");

    network.loadCheckpoint(fin);

    int correct = 0;
    for (int i = 0; i < test_image.size(); i++) {
        Mat result = network.forward(test_image[i]);
        // cout << "result: " << max_element(result[0], result[0] + 10) - result[0] << "  ";
        // cout << "answer: " << test_label[i] << endl;
        if (max_element(result[0], result[0] + 10) - result[0] == test_label[i]) {
            correct++;
        }
    }
    cout << "accuracy on test dataset: " << correct / (float)test_image.size() << endl;

    cout << "forward time: " << network.forwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "backward time: " << network.backwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication time: " << mutil::multiplyTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication count: " << mutil::multiplyCount << endl;
    cout << "matrix construct time:" << mutil::constructTime / (float)CLOCKS_PER_SEC << endl;
}

int main(void)
{
    cin.tie(0);
    // vector<float> v(25 * 3);
    // for (int j = 0; j < 3; j++)
    //     for (int i = 0; i < 25; i++) {
    //         v[j * 25 + i] = (j + 1) * 100 + i + 1;
    //     }
    // Mat img(3, 25, v);
    // auto osize = mutil::compute_output_size(5, 5, 2, 2, 1, 0);
    // Mat col(3, osize.first * osize.second * 4);
    // mutil::im2col(img, 3, 5, 5, { 2, 2 }, 1, 0, col);
    // vector<float> v2(3 * 4);
    // for (int j = 0; j < 3; j++)
    //     for (int i = 0; i < 4; i++)
    //         v2[j * 4 + i] = i + 1;
    // Mat kernel(3, 2 * 2, v2);
    // Mat ret(1, osize.first * osize.second);
    // Mat convret(osize.first, osize.second);
    // for (int i = 0; i < 3; i++) {
    //     Kernel in(osize.first * osize.second, 2 * 2, col[i]);
    //     Kernel k(4, 1, kernel[i]);
    //     Kernel out(osize.first * osize.second, 1, ret[0]);
    //     mutil::multiply(in, k, out);
    //     Kernel convimg(5, 5, img[i]);
    //     Kernel conk(2, 2, kernel[i]);
    //     Kernel convout(osize.first, osize.second, convret[0]);
    //     mutil::conv(convimg, conk, convout, 1, 0);
    //     for (int j = 0; j < osize.first; j++) {
    //         for (int k = 0; k < osize.second; k++) {
    //             cout << convret[0][j * osize.first + k] << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    //     for (int j = 0; j < osize.first; j++) {
    //         for (int k = 0; k < osize.second; k++) {
    //             cout << ret[0][j * osize.first + k] << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }

    train();
}