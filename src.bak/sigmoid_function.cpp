#include "sigmoid_function.hpp"

double sigmoid_function::activate(double value)
{
	return 1.0 / (1.0 + std::exp(-value));
}
double sigmoid_function::activate_differential(double value)
{
	double temp = activate(value);
	return (1.0 - temp) * temp;
}

