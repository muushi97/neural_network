#ifndef IG_INITIALIZER_HPP
#define IG_INITIALIZER_HPP

#include <random>

#include "perceptron.hpp"

namespace Raise_the_FLAG
{
	class initializer
	{
	private:

	public:
		// 初期化
		void initialize(perceptron &Network)
		{
			std::random_device random;
			std::mt19937 engine(random());

			// (平均, 標準偏差)
			std::normal_distribution<> n_dist(0.0, 1.0);

			// (最小値, 最大値)
			std::uniform_real_distribution<> ur_dist(-1.0, 1.0);

			// 閾値の初期化
			for (unsigned int i = 2; i <= Network.NetworkSize(); ++i)
			{
				for (unsigned int j = 1; j <= Network.LayerSize(i); ++j)
				{
					Network.Threshold(i, j) = ur_dist(engine);
				}
			}

			// 重みの初期化
			for (unsigned int i = 1; i <= Network.NetworkSize() - 1; ++i)
			{
				for (unsigned int j = 1; j <= Network.LayerSize(i); ++j)
				{
					for (unsigned int k = 1; k <= Network.LayerSize(i + 1); ++k)
					{
						Network.Weight(i, j, k) = ur_dist(engine);
					}
				}
			}
		}
	};
}

#endif
