#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

#include "trainer.hpp"
#include "loader.hpp"

using namespace std;

int main() {
    // network
    network<double>      net{ new fully_connected_layer<double>(2, 2)
                            , new sigmoid_layer<double>(2)
                            , new fully_connected_layer<double>(2, 1)
                            , new sigmoid_layer<double>(1)
                            };

    tensor<double> temp1{2}, temp2{1};

    std::unique_ptr<loader<double>> ldr(new mnist_loader<double>( "../../dataset/mnist/train-images.idx3-ubyte"
                                                                , "../../dataset/mnist/train-labels.idx1-ubyte"
                                                                , "../../dataset/mnist/t10k-images.idx3-ubyte"
                                                                , "../../dataset/mnist/t10k-labels.idx1-ubyte"
                                                                , 0.0, 1.0, 0.0, 0.0, 1));

    {
        int i = 0;
        do {
            std::cout << "train size : " << ldr->train_size() << std::endl;
            std::cout << "test size : " << ldr->test_size() << std::endl;
            std::cout << "valid size : " << ldr->valid_size() << std::endl;
            std::cin >> i;
            if (i < 0) break;

            tensor<double> d = ldr->get_valid_data(i);
            tensor<double> l = ldr->get_valid_label(i);

            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    std::cout << (d(x + 28 * y) < 0.5 ? "00" : "11");
                    if (d(x + 28 * y) < 0 || d(x + 28 * y) > 1.0) std::cout << std::endl << "error" << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << "label : ";
            for (int i = 0; i < l.dim(0); i++) {
                std::cout << (l(i) < 0.5 ? 0 : 1) << ", ";
            }
            std::cout << std::endl << std::endl;
        } while(i >= 0);
    }

    // train data
    std::vector<tensor<double>> xs, ts;
    temp1(0) = 0.0; temp1(1) = 0.0;   temp2(0) = 0.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 0.0; temp1(1) = 1.0;   temp2(0) = 1.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 1.0; temp1(1) = 0.0;   temp2(0) = 1.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 1.0; temp1(1) = 1.0;   temp2(0) = 0.0;
    xs.push_back(temp1); ts.push_back(temp2);


    trainer<double> tr;

    // initialize
    tr.set_parameter_by_uniform_distribution(net, -0.4, 0.4);
    std::size_t N = 100000;
    double eta = 0.1;

    // train
    for (int i = 0; i < N; i++)
        tr.parameter_update(net, eta, xs, ts);

    // result
    tensor<double> y;
    y = net.propagate(xs[0]);
    std::cout << y(0) << std::endl;
    y = net.propagate(xs[1]);
    std::cout << y(0) << std::endl;
    y = net.propagate(xs[2]);
    std::cout << y(0) << std::endl;
    y = net.propagate(xs[3]);
    std::cout << y(0) << std::endl;

    return 0;
}

