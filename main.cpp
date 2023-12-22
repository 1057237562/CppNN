#include "mnist_loader.cpp"
#include "network.cpp"
#include "optimizer.cpp"
#include <iostream>
#include <time.h>
#include <vector>
#define endl '\n'
using namespace std;
using namespace mutil;

int main(void)
{
    cin.tie(0);

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

    Network network({ new FlattenLayer(28, 28), new FullyConnectedLayer(28 * 28, 16), new SigmoidLayer(16), new FullyConnectedLayer(16, 16), new SigmoidLayer(16), new FullyConnectedLayer(16, 10), new SigmoidLayer(10) }, new SDG(train_data, 0.5, 10));
    network.init();
    network.train(1);

    for (int i = 0; i < 10; i++) {
        Mat result = network.forward(test_image[i]);
        cout << "result: " << max_element(result[0], result[0] + 10) - result[0] << "  ";
        cout << "answer: " << test_label[i] << endl;
    }

    cout << "forward time: " << network.forwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "backward time: " << network.backwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication time: " << mutil::multiplyTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication count: " << mutil::multiplyCount << endl;
    cout << "matrix construct time:" << mutil::constructTime / (float)CLOCKS_PER_SEC << endl;
    return 0;
}