#ifndef IG_RAMP_FUNCTION_HPP
#define IG_RAMP_FUNCTION_HPP

#include "_activation_function.hpp"

namespace Raise_the_FLAG
{
	// ランプ関数
	class ramp_function : public _activation_function
	{
	public:
		virtual double activate(double value);
		virtual double activate_differential(double value);
	};
}

#endif
