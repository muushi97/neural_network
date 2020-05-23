#ifndef IG_LAYER_HPP
#define IG_LAYER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

#include "tensor.hpp"

// layer
template <class T>
class base_layer {
    std::vector<std::size_t> input_size;
    std::vector<std::size_t> output_size;

public:
    const std::size_t inputSize(std::size_t i) const { return input_size[i]; }
    const std::vector<std::size_t> &inputSize() const { return input_size; }
    const std::size_t inputRank() const { return input_size.size(); }
    const std::size_t outputSize(std::size_t i) const { return output_size[i]; }
    const std::vector<std::size_t> &outputSize() const { return output_size; }
    const std::size_t outputRank() const { return output_size.size(); }

    // コンストラクタ
    base_layer(std::initializer_list<std::size_t> is, std::initializer_list<std::size_t> os) : input_size(is), output_size(os) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const = 0;
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dx) const = 0;

    // パラメータへのアクセス
    virtual tensor<T> &parameter() = 0;
    virtual const tensor<T> &parameter() const = 0;

    // デストラクタ
    virtual ~base_layer() { }
};
// 全結合層
template <class T>
class fully_connected_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> weight;

public:
    // コンストラクタ
    fully_connected_layer(std::size_t is, std::size_t os) : base_layer<T>{{is}, {os}}, weight{is+1, os} { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int o = 0; o < base::outputSize(0); o++) {
            y(o) = weight(base::inputSize(0), o); // しきい値で初期化
            for (int i = 0; i < base::inputSize(0); i++) {
                y(o) += x(i) * weight(i, o);
            }
        }

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize(0) };
        tensor<T> dw{ base::inputSize(0) + 1, base::outputSize(0) };

        for (int i = 0; i < base::inputSize(0); i++) {
            dx(i) = 0;
            for (int o = 0; o < base::outputSize(0); o++) {
                dx(i) += dy(o) * weight(i, o);
                dw(i, o) = dy(o) * x(i);
            }
        }
        for (int o = 0; o < base::outputSize(0); o++)
            dw(base::inputSize(0), o) = dy(o);

        return { dx, dw };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return weight; }
    virtual const tensor<T> &parameter() const { return weight; }

    // デストラクタ
    virtual ~fully_connected_layer() { }
};
// sigmoid 関数
template <class T>
class sigmoid_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> alpha;
    T sig(T a, T x) const {
        //std::cout << "sigmoid : " << "x(" << x << ") -> sig(" << 1.0 / (1.0 + std::exp(-1.0 * a * x)) << ")" << std::endl;
        return 1.0 / (1.0 + std::exp(-1.0 * a * x));
    }

public:
    // コンストラクタ
    sigmoid_layer(std::size_t s) : base_layer<T>({s}, {s}), alpha({s}) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int i = 0; i < base::inputSize(0); i++) {
            y(i) = sig(1.0, x(i));
        }

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize() };
        tensor<T> da{ base::inputSize() };

        for (int i = 0; i < base::inputSize(0); i++) {
            T s = sig(1.0, x(i));
            //T s = sig(alpha(i), x(i));
            dx(i) = 1.0      * s * (1.0 - s) * dy(i);
            //dx(i) = alpha(i) * s * (1.0 - s);
            da(i) = x(i)     * s * (1.0 - s) * dy(i);
        }

        return { dx, da };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return alpha; }
    virtual const tensor<T> &parameter() const { return alpha; }

    // デストラクタ
    virtual ~sigmoid_layer() { }
};
// ReLU 関数
template <class T>
class ReLU_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> alpha;

public:
    // コンストラクタ
    ReLU_layer(std::size_t s) : base_layer<T>({s}, {s}), alpha({s}) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int i = 0; i < base::inputSize(0); i++)
            y(i) = x(i) > 0.0 ? x(i) : 0.0;

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize() };

        for (int i = 0; i < base::inputSize(0); i++)
            dx(i) = x(i) > 0.0 ? 1.0 : -0.1;

        return { dx, tensor<T>{0} };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return alpha; }
    virtual const tensor<T> &parameter() const { return alpha; }

    // デストラクタ
    virtual ~ReLU_layer() { }
};

// レイヤーの別名
template <class T>
using layer = base_layer<T>;

#endif

