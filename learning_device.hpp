#ifndef IG_LEARNING_DEVICE_HPP
#define IG_LEARNING_DEVICE_HPP

#include <vector>

namespace Raise_the_FLAG
{
	class perceptron;
	class perceptron_parameter;

	class learning_device
	{
	private:
		double m_LearningCoefficient;

	public:
		// コンストラクタ
		learning_device(double LearningCoefficient);

		// 学修係数のセット
		void setLearningCoefficient(double LearningCoefficient);

		// 重み，及び閾値の更新料を計算
		void calculate_difference(perceptron &network, perceptron_parameter &parameter, std::vector<double> InputSignal, std::vector<double> TeacherSignal);

		// 更新量分，更新する
		void learn(perceptron &network, perceptron_parameter &parameter);

	};
}

#endif
