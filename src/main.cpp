#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>
#include <iomanip>

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


void GPR_test() {
    std::random_device seed_gen;
    std::mt19937_64 engine(seed_gen());
    std::normal_distribution<> dist(0.0, 0.2);
    //GPR gpr(0.2, GPR::se);
    GPR gpr(0.2, GPR::m52);

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
        //std::cout << "E = " << ev[0] << ", " << "V = " << ev[1] << "(" << std::sqrt(ev[1]) << ")" << std::endl;
        std::cout << x(0) << " " << ev[0] << " " << std::sqrt(ev[1]) << std::endl;
        //khstd::cout << x(0) << " " << ev[0] << " " << ev[1]) << std::endl;
    }
}


int expe1(int argc, char *argv[]) {
    double eta = 0.007;
    std::size_t batch_size = 128 * 2;
    std::size_t N = 500;
    double valid_rate = 0.20;
    double out_min = 0.1, out_max = 0.9;
    double weight_uniform_left = -0.4, weight_uniform_right = 0.4;
    double weight_normal_mean = 0.0, weight_normal_variance = 1.0;
    bool uniform_flag = true;
    std::vector<std::string> model = { "784", "sigm", "30", "sigm", "10" };
    network<double>      net{ new fully_connected_layer<double>(784, 30)
                            , new sigmoid_layer<double>(30)
                            , new fully_connected_layer<double>(30, 10)
                            , new sigmoid_layer<double>(10)
                            };
    std::size_t h = 20;
    std::string prename = "test";
    std::string postname = "test";
    std::string args = "";
    for (int i = 0; i < argc; i++)
        args += std::string(argv[i]) + (i < argc-1 ? " " : "");

    // initialize-1
    {
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-lr") { // learning rate
                eta = std::stod(std::string(argv[i+1])); i++;
            }
            else if (arg == "-bs") { // batch size
                batch_size = std::stoi(std::string(argv[i+1])); i++;
            }
            else if (arg == "-w") { // 重みの初期値 (uniform or normal)
                std::string dist = std::string(argv[i+1]); i++;
                if (dist == "uni") {
                    uniform_flag = true;
                    weight_uniform_left = std::stod(std::string(argv[i+1])); i++;
                    weight_uniform_right = std::stod(std::string(argv[i+1])); i++;
                }
                else if (dist == "norm") {
                    uniform_flag = false;
                    weight_normal_mean = std::stod(std::string(argv[i+1])); i++;
                    weight_normal_variance = std::stod(std::string(argv[i+1])); i++;
                }
                else return -1;
            }
            else if (arg == "-es") { // epoch size
                N = std::stoi(std::string(argv[i+1])); i++;
            }
            else if (arg == "-m") { // 構成
                std::size_t LN = std::stoi(std::string(argv[i+1])); i++;
                std::size_t p, n;
                std::vector<layer<double>*> il(2 * (LN-1));
                model.resize(2 * (LN-1) + 1);
                model[0] = std::string(argv[i+1]); i++;
                p = std::stoi(model[0]);
                for (int j = 1; j < LN; j++) {
                    model[2*(j-1) + 1] = std::string(argv[i+1]); i++;
                    if (model[2*(j-1) + 1] == "sigm" || model[2*(j-1) + 1] == "relu") {
                        model[2*(j-1) + 2] = std::string(argv[i+1]); i++;
                        n = std::stoi(model[2*(j-1) + 2]);
                        il[2*(j-1)    ] = new fully_connected_layer<double>(p, n);
                        if (model[2*(j-1) + 1] == "sigm") il[2*(j-1) + 1] = new sigmoid_layer<double>(n);
                        else                              il[2*(j-1) + 1] = new ReLU_layer<double>(n);
                        p = n;
                    }
                    else return -1;
                }
                net = network<double>(il);
            }
            else if (arg == "-vd") { // valid data
                N = std::stod(std::string(argv[i+1])); i++;
            }
            else if (arg == "-or") { // output range
                out_min = std::stod(std::string(argv[i+1])); i++;
                out_max = std::stod(std::string(argv[i+1])); i++;
            }
            else if (arg == "-ep") { // 誤差の値を出力するタイミング
                h = std::stoi(std::string(argv[i+1])); i++;
            }
            else if (arg == "-o") { // 出力ファイル接頭辞
                prename = std::string(argv[i+1]); i++;
                postname = std::string(argv[i+1]); i++;
            }
            else return -1;
        }

        std::cerr << "command line arguments : " << args << std::endl;
        std::cerr << "learning rate : " << eta << std::endl;
        std::cerr << "batch size : " << batch_size << std::endl;
        std::cerr << "N : " << N << std::endl;
        std::cerr << "valid rate : " << valid_rate << std::endl;
        std::cerr << "out [min, max] : [" << out_min << ", " << out_max << "]" << std::endl;
        std::cerr << "mode : ";
        for (int i = 0; i < model.size(); i++)
            std::cerr << model[i] << " \n"[i==model.size()-1];
        if (uniform_flag)
            std::cerr << "weight -> uniform distribution : [" << weight_uniform_left << ", " << weight_uniform_right << "]" << std::endl;
        else
            std::cerr << "weight -> uniform distribution : mean=" << weight_normal_mean << ", var=" << weight_normal_variance << std::endl;
        std::cerr << "step size : " << h << std::endl;
        std::cerr << "out put file name prefix : \"" << prename << "\"" << std::endl;
        std::cerr << "out put file name postfix : \"" << postname << "\"" << std::endl;
    }


    // initialize-2
    std::unique_ptr<loader<double>> ldr(new mnist_loader<double>( "../../dataset/mnist/train-images.idx3-ubyte"
                                                                , "../../dataset/mnist/train-labels.idx1-ubyte"
                                                                , "../../dataset/mnist/t10k-images.idx3-ubyte"
                                                                , "../../dataset/mnist/t10k-labels.idx1-ubyte"
                                                                , out_min, out_max, valid_rate, 0.0, 1));
    std::vector<tensor<double>> xs(batch_size), ts(batch_size);
    trainer<double> tr;
    if (uniform_flag)
        tr.set_parameter_by_uniform_distribution(net, weight_uniform_left, weight_uniform_right);
    else
        tr.set_parameter_by_uniform_distribution(net, weight_normal_mean, weight_normal_variance);
    // エポックに応じた accuracy の履歴
    std::ofstream train_accuracy(prename + "-train-accuracy-" + postname + ".csv");
    std::ofstream valid_accuracy(prename + "-valid-accuracy-" + postname + ".csv");
    std::ofstream test_accuracy(prename + "-test-accuracy-" + postname + ".csv");
    // エポックに応じた error の履歴
    std::ofstream train_error(prename + "-train-error-" + postname + ".csv");
    std::ofstream valid_error(prename + "-valid-error-" + postname + ".csv");
    std::ofstream test_error(prename + "-test-error-" + postname + ".csv");
    // 書式
    train_accuracy << std::fixed << std::setprecision(10) << std::endl;
    train_error    << std::fixed << std::setprecision(10) << std::endl;
    valid_accuracy << std::fixed << std::setprecision(10) << std::endl;
    valid_error    << std::fixed << std::setprecision(10) << std::endl;
    test_accuracy  << std::fixed << std::setprecision(10) << std::endl;
    test_error     << std::fixed << std::setprecision(10) << std::endl;


    // train
    std::random_device seed_gen;
    std::mt19937_64 engine(seed_gen());
    std::uniform_int_distribution<> dist_train(0, ldr->train_size()-1);
    std::uniform_int_distribution<> dist_valid(0, ldr->valid_size()-1);
    for (int i = 0; i <= N; i++) {
        if (i % h == 0) std::cerr << "m.step : " << i;

        if (i != 0) {
            for (int j = 0; j < batch_size; j++) {
                int k = dist_train(engine);
                xs[j] = ldr->get_train_data(k);
                ts[j] = ldr->get_train_label(k);
            }
            tr.parameter_update(net, eta, xs, ts);
        }

        if (i % h == 0) {
            std::size_t counter;
            double tr_acc, v_acc, te_acc;
            double tr_e, v_e, te_e;

            tr_e = 0.0;
            counter = 0;
            for (int j = 0; j < ldr->train_size(); j++) {
                tensor<double> y = net.propagate(ldr->get_train_data(j));
                tensor<double> z = ldr->get_train_label(j);
                tr_e += tr.mse(y, z);
                counter += max_1rand_tensor(y) == max_1rand_tensor(z) ? 1 : 0;
            }
            tr_acc = static_cast<double>(counter) / static_cast<double>(ldr->train_size());

            v_e = 0.0;
            counter = 0;
            for (int j = 0; j < ldr->valid_size(); j++) {
                tensor<double> y = net.propagate(ldr->get_valid_data(j));
                tensor<double> z = ldr->get_valid_label(j);
                v_e += tr.mse(y, z);
                counter += max_1rand_tensor(y) == max_1rand_tensor(z) ? 1 : 0;
            }
            v_acc = static_cast<double>(counter) / static_cast<double>(ldr->valid_size());

            te_e = 0.0;
            counter = 0;
            for (int j = 0; j < ldr->test_size(); j++) {
                tensor<double> y = net.propagate(ldr->get_test_data(j));
                tensor<double> z = ldr->get_test_label(j);
                te_e += tr.mse(y, z);
                counter += max_1rand_tensor(y) == max_1rand_tensor(z) ? 1 : 0;
            }
            te_acc = static_cast<double>(counter) / static_cast<double>(ldr->test_size());

            std::cerr << ",  error(train, valid, test) : " << tr_e << ", " << v_e << ", " << te_e;
            std::cerr << ",  accuracy(train, valid, test) : " << tr_acc << ", " << v_acc << ", " << te_acc << std::endl;
            train_accuracy << i << " " << tr_acc << std::endl;
            train_error    << i << " " << tr_e   << std::endl;
            valid_accuracy << i << " " <<  v_acc << std::endl;
            valid_error    << i << " " <<  v_e   << std::endl;
            test_accuracy  << i << " " << te_acc << std::endl;
            test_error     << i << " " << te_e   << std::endl;
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
        std::cerr << "test accuracy : " << static_cast<double>(counter) / static_cast<double>(ldr->test_size()) << std::endl;
    }
    //int i = 0;
    //tensor<double> y;
    //do {
    //    std::cerr << ">> ";
    //    std::cin >> i;
    //    if (i < 0) break;
    //    y = net.propagate(xs[i]);
    //    for (int j = 0; j < y.dim(0); j++) {
    //        std::cerr << j << " : " << std::round(y(j) * 100) << "(" << std::round(ts[i](j) * 100) << ")" << std::endl;
    //    }
    //    if (max_1rand_tensor(y) == max_1rand_tensor(ts[i])) std::cerr << "correct" << std::endl;
    //    else                                                std::cerr << "not correct" << std::endl;
    //    std::cerr << std::endl;
    //} while(true);


    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        test_xor();
    }
    else {
        std::string a(argv[1]);
        if (a == "xor") test_xor();
        if (a == "gpr") GPR_test();
        if (a == "expe1") return expe1(argc, argv);
    }

    return 0;
}

