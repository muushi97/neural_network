#ifndef IG_FUNCTIONS_HPP
#define IG_FUNCTIONS_HPP

#include "tensor.hpp"

template <class T, class U> class fully_connection;
template <class T, std::size_t... Ns, class U, std::size_t... Ms>
class fully_connection<tensor<T, Ns...>, tensor<U, Ms...>> {
    tensor<T, Ns..., Ms...> w;
    tensor<T, Ms...> b;

public:
    fully_connection() : w() { }
    tensor<U, Ms...> product(const tensor<T, Ns...> &x) const {
        tensor<U, Ms...> out;
        // 伝播
    }
};

template <class T, class U> class max;
template <class T, std::size_t... Ns, class U>
class max<tensor<T, Ns...>, tensor<U>> {
public:
    max() { }
    tensor<U> product(const tensor<T, Ns...> &x) {
        // 伝播
    }
};

class pooling;
class convolution;
class sigmoid;
class ReLU;

#endif
