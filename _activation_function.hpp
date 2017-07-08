#ifndef IG__ACTIVATION_FUNCTION_HPP
#define IG__ACTIVATION_FUNCTION_HPP

#include <cmath>
#include <functional>
#include <valarray>

namespace Raise_the_FLAG
{
	class _activation_function
	{
	public:
		virtual double activate(double value);
		virtual double activate_differential(double value);
	};
}

#endif
