#include <cstdio>
#include <iostream>

#include "perceptron.hpp"
#include "perceptron_parameter.hpp"

#include "initializer.hpp"

#include "learning_device.hpp"

#include "activation_function.hpp"

namespace Raise_the_FLAG
{
	void test()
	{
		// 活性化関数を定義
		sigmoid_function sigmoid;	// シグモイド関数
		ramp_function ramp;			// ランプ関数
		softplus_function softplus;	// ソフトプラス関数

		// ネットワークの形状と活性化関数を渡して，ネットワークを生成
		perceptron net({2,3,3,1} , &softplus);

		// ネットワーク初期化オブジェクトを生成
		initializer init;

		// ネットワークの初期化
		init.initialize(net);

		// 出力値を表示
		//std::cout << "0 0 , " << net({0.0,0.0})[0] << std::endl;
		//std::cout << "0 1 , " << net({0.0,1.0})[0] << std::endl;
		//std::cout << "1 0 , " << net({1.0,0.0})[0] << std::endl;
		//std::cout << "1 1 , " << net({1.0,1.0})[0] << std::endl;

		// 学習装置と変化量オブジェクトを生成
		learning_device testament(0.5);
		perceptron_parameter hoge({2,3,3,1});

		// 学習
		std::random_device random;
		std::mt19937 engine(random());
		// (最小値, 最大値)
		std::uniform_int_distribution<> ui_dist(0 , 3);
		for (unsigned int i = 0; i < 4096; ++i)
		{
			switch (ui_dist(engine))
			{
			case 0:
				testament.calculate_difference(net , hoge , {0.0,0.0} , {0.0});
				break;

			case 1:
				testament.calculate_difference(net , hoge , {0.0,1.0} , {1.0});
				break;

			case 2:
				testament.calculate_difference(net , hoge , {1.0,0.0} , {1.0});
				break;

			case 3:
				testament.calculate_difference(net , hoge , {1.0,1.0} , {0.0});
				break;
			}
			testament.learn(net , hoge);
		}

		// 出力値を表示
		std::cout << "0 0 , " << net({0.0,0.0})[0] << std::endl;
		std::cout << "0 1 , " << net({0.0,1.0})[0] << std::endl;
		std::cout << "1 0 , " << net({1.0,0.0})[0] << std::endl;
		std::cout << "1 1 , " << net({1.0,1.0})[0] << std::endl;
		std::cout << "error :  " <<
			(pow((net({0.0,0.0})[0] - 1.0) , 2)
			+ pow((net({1.0,0.0})[0] - 0.0) , 2)
			+ pow((net({0.0,1.0})[0] - 0.0) , 2)
			+ pow((net({1.0,1.0})[0] - 1.0) , 2))
			* 0.5
			<< std::endl;
	}
}

int main()
{/**/
	while (1)
	{
		Raise_the_FLAG::test();

		if (!(getchar()))
			break;
	}/**/

	// げっちゃぁ
	getchar();

	return 0;
}
