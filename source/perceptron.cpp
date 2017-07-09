#include "perceptron.hpp"

using namespace Raise_the_FLAG;

#define base perceptron_parameter

// コンストラクタ
perceptron::perceptron(std::vector<unsigned int> NetworkForm, _activation_function *ActivationFunction) : base(NetworkForm)
{
	m_ActivationFunction = ActivationFunction;

	unsigned int Size = base::NetworkSize();

	m_InputSignal.resize(base::LayerSize(1));
	m_OutputSignal.resize(base::LayerSize(Size));
}

// 入力
void perceptron::input(std::vector<double> InputSignal)
{/*
 InputSignal を入力信号として，1層目の各ニューロンに書き込む

 配列 InputSignal の長さは，1層目のニューロンの数以上であるとする．もし，配列 InputSignal の長さが，1層目のニューロンの数よりも大きい場合は余った部分を無視する．
 */
	for (unsigned int i = 1; i <= LayerSize(1); ++i)
	{
		m_InputSignal[i - 1] = InputSignal[i - 1];
		base::NeuronOutput(1, i) = InputSignal[i - 1];
	}
}
// 順伝播
void perceptron::propagate()
{/*
 現在の1層目(入力層)に入力されている値を伝播させる．
 */
	for (unsigned int i = 2; i <= base::NetworkSize(); ++i)
	{
		for (unsigned int j = 1; j <= base::LayerSize(i); ++j)
		{
			base::NeuronValue(i , j) = base::Threshold(i , j);
			for (unsigned int k = 1; k <= base::LayerSize(i - 1); ++k)
			{
				base::NeuronValue(i , j) += base::NeuronOutput(i - 1 , k) * base::Weight(i - 1 , k , j);
			}
			base::NeuronOutput(i , j) = m_ActivationFunction->activate(base::NeuronValue(i , j));
		}
	}

	unsigned int LastLayer = NetworkSize();
	unsigned int Size = LayerSize(LastLayer);
	for (unsigned int i = 1; i <= Size; ++i)
	{
		m_OutputSignal[i - 1] = base::NeuronOutput(LastLayer , i);
	}
}
// 出力
const std::vector<double> &perceptron::output() const
{/*
 最終層(出力層)を定数ベクトルとして返す
 */
	return m_OutputSignal;
}

// 入力，順伝播，出力をまとめて行う
const std::vector<double> &perceptron::operator() (std::vector<double> InputSignal)
{
	input(InputSignal);
	propagate();
	return output();
}

// ネットワークのサイズ(層数)を返す
unsigned int perceptron::NetworkSize() const
{
	return base::NetworkSize();
}
// ある層のサイズ(ニューロン数)を返す
unsigned int perceptron::LayerSize(unsigned int LayerNumber) const
{/*
 LayerNumber層目のニューロンの数を返す

 LayerNumber はパーセプトロンの層数を最大値，最小値を 1 とする．
 */
	return base::LayerSize(LayerNumber);
}
