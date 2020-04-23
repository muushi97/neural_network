#ifndef IG__ACTIVATION_FUNCTION_HPP
#define IG__ACTIVATION_FUNCTION_HPP

#include <cmath>
#include <functional>

namespace hoge {
    template <class T>
    T sigmoid(T u) {
        return 1.0 / (1.0 + std::exp(-u));
    }
    template <class T>
    T softplus(T u) {
        return std::log(1.0 + std::exp(u));
    }
    template <class T>
    T ramp(T u) {
        return u > 0.0 ? u : 0.0;
    }
//double ramp_function::activate_differential(double value)
//{
//	if (value == 0.0)
//		return 0.5;
//	return value > 0.0 ? 1.0 : 0.0;
//}
}

#endif

