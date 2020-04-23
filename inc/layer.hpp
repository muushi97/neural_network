#ifndef IG_LAYER_HPP
#define IG_LAYER_HPP

#include <variant>
#include <functional>
#include "vector.hpp"

namespace hoge {
    template <class I, class O>
    class based_layer {
        int input_size;
        int output_size;
    protected:
        int inputSize() const { return input_size; }
        int outputSize() const { return output_size; }
    public:
        based_layer(int is, int os) : input_size(is), output_size(os) { }
    };

    template <class I, class O>
    class fully_connected_layer : public based_layer<I, O> {
        tensor<I> weight;
    public:
        fully_connected_layer(int is, int os) : based_layer<I, O>(is, os), weight({is, os}) { }
        tensor<O> propagate(const tensor<I> x) const {
            // inputSize() == x.size() == weight.size() && forall i, outputSize () == weight[i].size()
            tensor<O> y(based_layer<I, O>::outputSize());// あと初期化

            for (int o = 0; o < based_layer<I, O>::outputSize(); o++) {
                y[o] = 0; // ゼロ元で初期化
                for (int i = 0; i < based_layer<I, O>::inputSize(); i++)
                    y[o] += x[i] * weight[i][o];
            }
            return y;
        }
    };

    template <class I, class O>
    class activation_layer : public based_layer<I, O> {
        std::function<O(I)> f;
    public:
        activation_layer(int is) : based_layer<I, O>(is, is) { }
        tensor<O> propagate(const tensor<I> x) const {
            tensor<O> y({based_layer<I, O>::outputSize()});
            for (int o = 0; o < based_layer<I, O>::outputSize(); o++) y[o] = f(x[o]);
            return y;
        }
    };

    template <class I, class O>
    using layer = std::variant<fully_connected_layer<I, O>, activation_layer<I, O>>;
    template <class T, class I, class O>
    tensor<O> propagate(const T &l, const tensor<I> x) { l.propagate(x); }

}

#endif

