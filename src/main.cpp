#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

#include "trainer.hpp"
#include "loader.hpp"
#include "gaussian_process_regression.hpp"

using namespace std;

template <class T>
std::size_t max_1rand_tensor(const tensor<T> &t) {
    std::size_t i = 0;
    for (int j = 1; j < t.dim(0); j++)
        if (t(i) < t(j))
            i = j;
    return i;
}

void test_xor() {
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
}

void test_mnist() {
    std::unique_ptr<loader<double>> ldr(new mnist_loader<double>( "../../dataset/mnist/train-images.idx3-ubyte"
                                                                , "../../dataset/mnist/train-labels.idx1-ubyte"
                                                                , "../../dataset/mnist/t10k-images.idx3-ubyte"
                                                                , "../../dataset/mnist/t10k-labels.idx1-ubyte"
                                                                , 0.1, 0.9, 1.0 / 6.0, 0.0, 1));

    int batch_size = 128 * 2;
    std::size_t N = 500;
    //double eta = 0.005;
    double eta = 0.007;
    int print_step = 20;

    // train data
    std::vector<tensor<double>> xs(batch_size), ts(batch_size);

    // network
    network<double>      net{ new fully_connected_layer<double>(784, 30)
                            , new sigmoid_layer<double>(30)
                            , new fully_connected_layer<double>(30, 10)
                            //, new sigmoid_layer<double>(392)
                            //, new fully_connected_layer<double>(392, 10)
                            , new sigmoid_layer<double>(10)
                            };


    trainer<double> tr;

    // initialize
    tr.set_parameter_by_uniform_distribution(net, -0.4, 0.4);

    // train
    std::random_device seed_gen;
    std::mt19937_64 engine(seed_gen());
    std::uniform_int_distribution<> dist_train(0, ldr->train_size()-1);
    std::uniform_int_distribution<> dist_valid(0, ldr->valid_size()-1);
    for (int i = 0; i < N; i++) {
        if (i % print_step == 0) std::cout << "m.step : " << i;

        for (int j = 0; j < batch_size; j++) {
            int k = dist_train(engine);
            xs[j] = ldr->get_train_data(k);
            ts[j] = ldr->get_train_label(k);
        }

        tr.parameter_update(net, eta, xs, ts);

        if (i % print_step == 0) {
            double e = 0.0;
            std::size_t counter = 0;
            for (int j = 0; j < ldr->valid_size(); j++) {
                tensor<double> y = net.propagate(ldr->get_valid_data(j));
                tensor<double> z = ldr->get_valid_label(j);
                e += tr.mse(y, z);
                counter += max_1rand_tensor(y) == max_1rand_tensor(z) ? 1 : 0;
            }
            std::cout << ",  error : " << e;
            std::cout << ",  acuracy : " << static_cast<double>(counter) / static_cast<double>(ldr->valid_size()) << std::endl;
        }
    }

    // result
    xs.clear(); ts.clear();
    xs.resize(ldr->test_size()), ts.resize(ldr->test_size());
    for (int i = 0; i < ldr->test_size(); i++) {
        xs[i] = ldr->get_test_data(i);
        ts[i] = ldr->get_test_label(i);
    }
    {
        double e = 0.0;
        std::size_t counter = 0;
        for (int j = 0; j < ldr->test_size(); j++) {
            tensor<double> y = net.propagate(xs[j]);
            e += tr.mse(y, ts[j]);
            counter += max_1rand_tensor(y) == max_1rand_tensor(ts[j]) ? 1 : 0;
        }
        std::cout << "test acuracy : " << static_cast<double>(counter) / static_cast<double>(ldr->test_size()) << std::endl;
    }
    int i = 0;
    tensor<double> y;
    do {
        std::cout << ">> ";
        std::cin >> i;
        if (i < 0) break;
        y = net.propagate(xs[i]);
        for (int j = 0; j < y.dim(0); j++) {
            std::cout << j << " : " << std::round(y(j) * 100) << "(" << std::round(ts[i](j) * 100) << ")" << std::endl;
        }
        if (max_1rand_tensor(y) == max_1rand_tensor(ts[i])) std::cout << "correct" << std::endl;
        else                                                std::cout << "not correct" << std::endl;
        std::cout << std::endl;
    } while(true);
}

void GPR_test() {
    std::random_device seed_gen;
    std::mt19937_64 engine(seed_gen());
    std::normal_distribution<> dist(0.0, 0.02);
    GPR gpr(0.02);
    //gpr.jordan_test();

    //std::cout << "train" << std::endl;
    int N;
    std::cout << "sumple number >> ";
    std::cin >> N;
    for (int i = 0; i < N; i++) {
        tensor<double> x{1}; x(0) = (3.14 * 2 / static_cast<double>(N-1)) * i;
        double y = std::sin(x(0)) + dist(engine);
        //std::cout << x(0) << ", " << y << std::endl;
        gpr.add_train_data(x, y);
    }

    //std::cout << std::endl << "test" << std::endl;
    //double i = 0.0;
    //do {
    //    std::cout << ">> ";
    //    std::cin >> i;
    //    tensor<double> x{1}; x(0) = i;
    //    auto ev = gpr.EV(x);
    //    std::cout << "E = " << ev[0] << ", " << "V = " << ev[1] << "(" << std::sqrt(ev[1]) << ")" << std::endl;
    //} while(i > 0.0);
    int NN = 100;
    for (int i = 0; i < NN; i++) {
        tensor<double> x{1}; x(0) = (3.14 * 2 / static_cast<double>(NN-1)) * i;
        auto ev = gpr.EV(x);
        std::cout << "E = " << ev[0] << ", " << "V = " << ev[1] << "(" << std::sqrt(ev[1]) << ")" << std::endl;
        //std::cout << x(0) << " " << ev[0] << " " << std::sqrt(ev[1]) << std::endl;
    }
}

int main() {
    //test_xor();
    //test_mnist();
    GPR_test();

    return 0;
}

