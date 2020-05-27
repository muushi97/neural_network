#ifndef IG_TRAINER_HPP
#define IG_TRAINER_HPP

#include "network.hpp"

// 学習機
template <class T>
class trainer {
public:
    // mean squared error
    double mse(const tensor<T> x, const tensor<T> t) {
        T e = 0;
        T c = 0;
        T y, u;
        for (auto ind = x.begin(); ind != x.end(); ind.next()) {
            y = std::pow(x.at(ind) - t.at(ind), 2) - c;
            u = e + y;
            c = (u - e) - y;
            e = u;
        }
        return e;
    }
private:
    tensor<T> d_mse(const tensor<T> x, const tensor<T> t) {
        tensor<T> dx = x;
        for (auto ind = x.begin(); ind != x.end(); ind.next()) {
            dx.at(ind) -= t.at(ind);
        }
        return dx;
    }

public:
    T error;
    T previous_error() const { return error; }

    void set_parameter_by_uniform_distribution(network<T> &net, T a, T b) {
        std::random_device seed_gen;
        std::mt19937_64 engine(seed_gen());

        std::uniform_real_distribution<> dist(a, b);

        for (int l = 0; l < net.layer_size(); l++) {
            for (auto ind = net.parameter(l).begin(); ind != net.parameter(l).end(); ind.next()) {
                net.parameter(l).at(ind) = dist(engine);
            }
        }
    }

    std::vector<tensor<T>> backpropagate(network<T> &net, const tensor<T> x, const tensor<T> t) {
        int N = net.layer_size();
        std::vector<tensor<T>> xs(N + 1);
        std::vector<tensor<T>> ws(N);

        // propagation
        xs[0] = x;
        for (int i = 0; i < N; i++) {
            xs[i + 1] = net.propagate(i, xs[i]);
        }

        // error
        error = mse(xs[N], t);
        tensor<T> dy = d_mse(xs[N], t);

        // backpropagation
        std::array<tensor<T>, 2> res;
        for (int i = 0; i < N; i++) {
            res = net.backpropagate(N-1-i, xs[N-1-i], dy);
            dy = res[0];
            ws[N-1-i] = res[1];
        }

        return ws;
    }
    void parameter_update(network<T> &net, double eta, const std::vector<tensor<T>> xs, const std::vector<tensor<T>> ts) {
        std::vector<tensor<T>> ws = backpropagate(net, xs[0], ts[0]);
        std::vector<tensor<T>> ws_;
        for (int i = 1; i < xs.size(); i++) {
            //if (i % 1 == 0) std::cout << "t:step : " << i << std::endl;
            ws_ = backpropagate(net, xs[i], ts[i]);
            for (int l = 0; l < ws_.size(); l++) {
                for (auto ind = ws[l].begin(); ind != ws[l].end(); ind.next()) {
                    ws[l].at(ind) += ws_[l].at(ind);
                }
            }
        }

        for (int l = 0; l < ws.size(); l++) {
            for (auto ind = ws[l].begin(); ind != ws[l].end(); ind.next()) {
                net.parameter(l).at(ind) = net.parameter(l).at(ind) - eta * ws[l].at(ind);
            }
        }
    }
};

#endif

