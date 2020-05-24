#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

#include "trainer.hpp"

using namespace std;

int main() {
    // network
    network<double>      net{ new fully_connected_layer<double>(2, 2)
                            , new sigmoid_layer<double>(2)
                            , new fully_connected_layer<double>(2, 1)
                            , new sigmoid_layer<double>(1)
                            };

    tensor<double> temp1{2}, temp2{1};

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

