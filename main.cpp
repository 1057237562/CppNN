#include "mnist_loader.cpp"
#include "network.cpp"
#include "optimizer.cpp"
#include <iostream>
#include <time.h>
#include <vector>

using namespace std;
using namespace mutil;

int main(void)
{
    vector<Mat> training_image = read_mnist_images(".");
    vector<int> training_label = read_mnist_labels(".");

    vector<pair<Mat, Mat>> training_data;
    for (int i = 0; i < training_image.size(); i++) {
        Mat mat(1, 10);
        mat[0][training_label[i]] = 1;
        training_data.push_back({ training_image[i], mat });
    }

    Network network({ new FlattenLayer(28, 28), new FullyConnectedLayer(28 * 28, 16), new SigmoidLayer(16), new FullyConnectedLayer(16, 16), new SigmoidLayer(16), new FullyConnectedLayer(16, 10), new SigmoidLayer(10) }, new SDG(training_data, 0.5, 10));
    network.init();
    network.train(1);

    for (int i = 0; i < 10; i++) {
        Mat result = network.forward(training_image[i]);
        cout << "result: " << max_element(result[0].val.begin(), result[0].val.end()) - result[0].val.begin() << "  ";
        cout << "answer: " << training_label[i] << endl;
    }

    cout << "forward time: " << network.forwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "backward time: " << network.backwardTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication time: " << mutil::multiplyTime / (float)CLOCKS_PER_SEC << endl;
    cout << "matrix multiplication count: " << mutil::multiplyCount << endl;
    return 0;
}