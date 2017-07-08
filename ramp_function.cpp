#include "ramp_function.hpp"

using namespace Raise_the_FLAG;

double ramp_function::activate(double value)
{
	return value > 0.0 ? value : 0.0;
}
double ramp_function::activate_differential(double value)
{
	if (value == 0.0)
		return 0.5;
	return value > 0.0 ? 1.0 : 0.0;
}
