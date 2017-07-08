#ifndef IG_SIGMOID_FUNCTION_HPP
#define IG_SIGMOID_FUNCTION_HPP

#include "_activation_function.hpp"

namespace Raise_the_FLAG
{
	// シグモイド関数
	class sigmoid_function : public _activation_function
	{
	public:
		virtual double activate(double value)
		{
			return 1.0 / (1.0 + std::exp(-value));
		}
		virtual double activate_differential(double value)
		{
			double temp = activate(value);
			return (1.0 - temp) * temp;
		}
	};
}

#endif
