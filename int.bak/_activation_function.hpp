#ifndef IG__ACTIVATION_FUNCTION_HPP
#define IG__ACTIVATION_FUNCTION_HPP

#include <cmath>
#include <functional>
#include <valarray>

class _activation_function
{
public:
    virtual double activate(double value) = 0;
    virtual double activate_differential(double value) = 0;
};

#endif

