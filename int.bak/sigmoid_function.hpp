#ifndef IG_SIGMOID_FUNCTION_HPP
#define IG_SIGMOID_FUNCTION_HPP

#include "_activation_function.hpp"

// シグモイド関数
class sigmoid_function : public _activation_function
{
public:
    virtual double activate(double value);
    virtual double activate_differential(double value);
};

#endif

