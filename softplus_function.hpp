#ifndef IG_SOFTPLUS_FUNCTION_HPP
#define IG_SOFTPLUS_FUNCTION_HPP

#include "_activation_function.hpp"

namespace Raise_the_FLAG
{
	// ソフトプラス関数
	class softplus_function : public _activation_function
	{
	public:
		virtual double activate(double value)
		{
			return std::log(1.0 + std::exp(value));
		}
		virtual double activate_differential(double value)
		{
			double e_x = std::exp(value);
			return e_x / (1.0 + e_x);
		}
	};
}

#endif
