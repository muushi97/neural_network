#ifndef IG_INITIALIZER_HPP
#define IG_INITIALIZER_HPP

#include <random>

namespace Raise_the_FLAG
{
	class perceptron;

	class initializer
	{
	private:
		std::mt19937 m_engine;
		double m_min, m_max;

	public:
		// 今ストラクた
		initializer();
		initializer(double min, double max);

		// 初期化
		void initialize(perceptron &Network);
	};
}

#endif
