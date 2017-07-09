#ifndef IG_SOFTPLUS_FUNCTION_HPP
#define IG_SOFTPLUS_FUNCTION_HPP

#include "_activation_function.hpp"

namespace Raise_the_FLAG
{
	// ソフトプラス関数
	class softplus_function : public _activation_function
	{
	public:
		virtual double activate(double value);
		virtual double activate_differential(double value);
	};
}

#endif
