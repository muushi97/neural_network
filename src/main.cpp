#include <iostream>

#include "dual_number.hpp"
#include "layer.hpp"

using namespace std;

int main() {

    hoge::layer<double, double> l = hoge::fully_connected_layer<double, double>(10, 2);
    hoge::layer<double, double> l2 = hoge::activation_layer<double, double>(10);
    std::vector<hoge::layer<double, double>> n{ hoge::fully_connected_layer<double, double>(1, 2)
                                              , hoge::activation_layer<double, double>(2)
                                              };

    hoge::tensor<double> x{0};

    for (int i = 0; i < n.size(); i++) {
        x = std::visit([&x](const auto& l){ return hoge::propagate(l, x); }, n[0]);
    }

    return 0;
}

