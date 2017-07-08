#ifndef IG_RAMP_FUNCTION_HPP
#define IG_RAMP_FUNCTION_HPP

#include "_activation_function.hpp"

namespace Raise_the_FLAG
{
	// ランプ関数
	class ramp_function : public _activation_function
	{
	public:
		virtual double activate(double value)
		{
			return value > 0.0 ? value : 0.0;
		}
		virtual double activate_differential(double value)
		{
			if (value == 0.0)
				return 0.5;
			return value > 0.0 ? 1.0 : 0.0;
		}
	};
}

#endif
