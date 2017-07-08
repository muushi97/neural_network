#include "learning_device.hpp"

#include "perceptron.hpp"
#include "perceptron_parameter.hpp"

using namespace Raise_the_FLAG;

// コンストラクタ
learning_device::learning_device(double LearningCoefficient)
{
	setLearningCoefficient(LearningCoefficient);
}

// 学修係数のセット
void learning_device::setLearningCoefficient(double LearningCoefficient)
{
	m_LearningCoefficient = LearningCoefficient;
}

// 重み，及び閾値の更新料を計算
void learning_device::calculate_difference(perceptron &network, perceptron_parameter &parameter, std::vector<double> InputSignal, std::vector<double> TeacherSignal)
{
	network.input(InputSignal);
	network.propagate();

	double ganma;

	unsigned int NetworkSize = network.NetworkSize();

	// 閾値更新
	for (unsigned int j = network.LayerSize(NetworkSize); j > 0; --j)
	{
		ganma = network.NeuronOutput(NetworkSize, j) - TeacherSignal[j - 1];
		parameter.Threshold(NetworkSize, j) = ganma * network.m_ActivationFunction->activate_differential(network.NeuronValue(NetworkSize, j));
	}
	for (unsigned int i = NetworkSize - 1; i > 1; --i)
	{
		for (unsigned int j = network.LayerSize(i); j > 0; --j)
		{
			ganma = 0.0;
			for (unsigned int k = network.LayerSize(i + 1); k > 0; --k)
			{
				ganma += parameter.Threshold(i + 1, k) * network.Weight(i, j, k);
			}

			parameter.Threshold(i, j) = ganma * network.m_ActivationFunction->activate_differential(network.NeuronValue(i, j));
		}
	}

	// 重み更新
	for (unsigned int i = NetworkSize - 1; i > 0; --i)
	{
		for (unsigned int j = network.LayerSize(i + 1); j > 0; --j)
		{
			for (unsigned int k = network.LayerSize(i); k > 0; --k)
			{
				parameter.Weight(i, k, j) = parameter.Threshold(i + 1, j) * network.NeuronOutput(i, k);
			}
		}
	}


}

// 更新量分，更新する
void learning_device::learn(perceptron &network, perceptron_parameter &parameter)
{
	// 閾値の更新
	for (unsigned int i = 2; i <= network.NetworkSize(); ++i)
	{
		for (unsigned int j = 1; j <= network.LayerSize(i); ++j)
		{
			network.Threshold(i, j) += -1.0 * m_LearningCoefficient * parameter.Threshold(i, j);
		}
	}

	// 重みの更新
	for (unsigned int i = 1; i <= network.NetworkSize() - 1; ++i)
	{
		for (unsigned int j = 1; j <= network.LayerSize(i); ++j)
		{
			for (unsigned int k = 1; k <= network.LayerSize(i + 1); ++k)
			{
				network.Weight(i, j, k) += -1.0 * m_LearningCoefficient * parameter.Weight(i, j, k);
			}
		}
	}
}
