#ifndef IG_NETWORK_HPP
#define IG_NETWORK_HPP

#include "layer.hpp"

template <class T>
class trainer;

// フィードフォワードネットワーク
template <class T>
class network {
    friend trainer<T>;

    std::vector<std::unique_ptr<layer<T>>> ls;

    // パラメータの取得
    tensor<T> &parameter(int i) {
        return ls[i]->parameter();
    }
    const tensor<T> &parameter(int i) const { return ls[i]->parameter(); }

    // 逆伝播
    std::array<tensor<T>, 2> backpropagate(int i, const tensor<T> x, const tensor<T> dy) { return ls[i]->backpropagate(x, dy); }

public:
    // コンストラクタ
    network(std::initializer_list<layer<T>*> il) {
        for (auto itr = il.begin(); itr != il.end(); itr++)
            ls.emplace_back(*itr);
    }

    // 順伝播
    tensor<T> propagate(int i, const tensor<T> x) { return ls[i]->propagate(x); }
    tensor<T> propagate(const tensor<T> x) {
        tensor<T> y = x;
        for (int i = 0; i < ls.size(); i++) {
            y = propagate(i, y);
        }
        return y;
    }

    std::size_t layer_size() const { return ls.size(); }
};

#endif

