#ifndef IG_LAYER_HPP
#define IG_LAYER_HPP

#include "tensor.hpp"

template <class T, std::size_t... Ns>
class layer {
    tensor<T, Ns...> unit;

public:
    layer() : unit() { }

    template <class P, class U, std::size_t... Ms>
    layer(const P &f, const tensor<U, Ms...> &x) : unit() { propagate(f, x); }

    template <class P, class U, std::size_t... Ms>
    void propagate(const P &w, const tensor<U, Ms...> &x) { unit = w(x); }

    void set(const tensor<T, Ns...> &x) { unit = x; }

    tensor<T, Ns...> &get() { return unit; }
    const tensor<T, Ns...> &get() const { return unit; }
};

#endif
