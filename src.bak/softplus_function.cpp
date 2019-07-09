#include "softplus_function.hpp"

double softplus_function::activate(double value)
{
	return std::log(1.0 + std::exp(value));
}
double softplus_function::activate_differential(double value)
{
	double e_x = std::exp(value);
	return e_x / (1.0 + e_x);
}

