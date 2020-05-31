#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>
#include <iomanip>
#include <sstream>

#include "trainer.hpp"
#include "loader.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    // network
    network<double>      net{ new fully_connected_layer<double>(1, 1)
                            , new sigmoid_layer<double>(1)
                            , new fully_connected_layer<double>(1, 1)
                            , new sigmoid_layer<double>(1)
                            };

    std::random_device seed_gen;
    std::mt19937_64 engine(seed_nge());

    std::size_t train_size = 100;
    std::size_t valid_size = 10;
    std::size_t test_size = 100;
    std::size_t N = 100000;
    std::size_t batch_size = 10;
    double eta = 0.1;
    double weight_left =   -0.4;
    double wweight_right = +0.4;

    std::vector<tensor<double>> train_xs(train_size), train_ts(train_size);
    std::vector<tensor<double>> valid_xs(valid_size), valid_ts(valid_size);
    std::vector<tensor<double>> test_xs(test_size), test_ts(test_size);
    /* generate data */ {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        tensor<double> temp1{1}, temp2{1};

        // train data
        for (std::size_t i = 0; i < 100; i++) {
            temp1(0) = dist(engine);
            temp2(0) = 10.0 * temp1(0);

            train_xs[i] = temp1;
            train_ts[i] = temp2;
        }

        // valid data
        for (std::size_t i = 0; i < 100; i++) {
            temp1(0) = dist(engine);
            temp2(0) = 10.0 * temp1(0);

            valid_xs[i] = temp1;
            valid_ts[i] = temp2;
        }

        // test data
        for (std::size_t i = 0; i < 100; i++) {
            temp1(0) = dist(engine);
            temp2(0) = 10.0 * temp1(0);

            test_xs[i] = temp1;
            test_ts[i] = temp2;
        }
    }

    // trainer
    trainer<double> tr;
    // initialize
    tr.set_parameter_by_uniform_distribution(net, weight_left, weight_right);

    /* train */ {
        std::uniform_int_distrubution<> dist(0, train_size);
        std::vector<tensor<double>> batch_xs(batch_size);
        std::vector<tensor<double>> batch_ts(batch_size);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < batch_size; j++) {
                int k = dist(engine);
                batch_xs[j] = train_xs[k];
                batch_ts[j] = train_ts[k];
            }
            tr.parameter_update(net, eta, batch_xs, batch_ts);
        }
    }

    /* test */ {
        double e = 0.0;
        std::size_t counter = 0;
        tensor<double> y;

        for (int i = 0; i < test_size; i++) {
            y = net.propagate(test_xs[i]);
        }
    }

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

