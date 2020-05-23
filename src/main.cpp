#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

#include "trainer.hpp"


using namespace std;

int main() {
    network<double>    n{ new fully_connected_layer<double>(2, 4)
                        , new sigmoid_layer<double>(4)
                        , new fully_connected_layer<double>(4, 1)
                        , new sigmoid_layer<double>(1)
                        };

    tensor<double> temp1{2}, temp2{1};

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

    tr.set_parameter_by_uniform_distribution(n, -0.4, 0.4);
    for (int i = 0; i < 100000; i++) {
        tr.parameter_update(n, 0.1, xs, ts);
        //getchar();
    }

    tensor<double> y;
    y = n.propagate(xs[0]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[1]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[2]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[3]);
    std::cout << y(0) << std::endl;

    return 0;
}

