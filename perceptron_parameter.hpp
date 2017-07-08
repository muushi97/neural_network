#ifndef IG_PERCEPTRON_PARAMETER_HPP
#define IG_PERCEPTRON_PARAMETER_HPP

#include <vector>
#include <valarray>

namespace Raise_the_FLAG
{
	class perceptron_parameter
	{
		friend class learning_device;

	private:
		std::vector<unsigned int> m_NetworkForm;
		std::vector<std::vector<double>> m_NeuronOutput;
		std::vector<std::vector<double>> m_NeuronValue;
		std::vector<std::vector<double>> m_Threshold;
		std::vector<std::vector<double>> m_Weight;

	protected:
		// あるニューロンの出力値を参照する
		double &NeuronOutput(unsigned int LayerNumber, unsigned int NeuronNumber)
		{/*
		 LayerNumber層目のNeuronNumber番目のニューロンの出力値にアクセスする．

		 LayerNumber はパーセプトロンの層数を最大値，最小値を 1 とする
		 NeuronNumber は LayerNumber層目のニューロンの数を最大値，最小値を 1 とする
		 */
			return m_NeuronOutput[LayerNumber - 1][NeuronNumber - 1];
		}
		// あるニューロンの値を参照する
		double &NeuronValue(unsigned int LayerNumber, unsigned int NeuronNumber)
		{/*
		 LayerNumber層目のNeuronNumber番目のニューロンの値にアクセスする．なお，1層目は入力層であり，値は存在しないためLayerNumber に 1 が入力されることは想定しない．

		 LayerNumber はパーセプトロンの層数を最大値，最小値を 2 とする
		 NeuronNumber は LayerNumber層目のニューロンの数を最大値，最小値を 1 とする
		 */
			return m_NeuronValue[LayerNumber - 2][NeuronNumber - 1];
		}
		// あるニューロンの閾値を参照する
		double &Threshold(unsigned int LayerNumber, unsigned int NeuronNumber)
		{/*
		 LayerNumber層目のNeuronNumber番目のニューロンの閾値にアクセスする．なお，1層目は入力そうであり，閾値は存在しないためLayerNumber に 1 が入力されることは想定しない．

		 LayerNumber はパーセプトロンの層数を最大値，最小値を 2 とする
		 NeuronNumber は LayerNumber層目のニューロンの数を最大値，最小値を 1 とする
		 */
			return m_Threshold[LayerNumber - 2][NeuronNumber - 1];
		}
		// あるニューロンと，あるニューロン間の重みを参照する
		double &Weight(unsigned int LayerNumber, unsigned int SourceNeuronNumber, unsigned int DistinationNeuronNumber)
		{/*
		 LayerNumber層目のSourceNeuronNumber番目のニューロンからLayerNumber+1層目のDistinationNeuronNumber番目のニューロン間の重みにアクセスする．

		 LayerNumber はパーセプトロンの層数-1を最大値，最小値を1とする．
		 SourceNeuronNumber は LayerNumber層目のニューロン数を最大値，最小値を 1 とする．
		 DistinationNeuronNumber は LayerNumber+1層目のニューロン数を最大値，最小値を 1 とする．
		 */
			SourceNeuronNumber--;
			DistinationNeuronNumber--;
			return m_Weight[LayerNumber - 1][DistinationNeuronNumber * LayerSize(LayerNumber) + SourceNeuronNumber];
		}

		// ネットワークのサイズ(層数)を返す
		unsigned int NetworkSize() const
		{/*
		 ネットワークの層の数を返す
		 */
			return m_NetworkForm.size();
		}
		// ある層のサイズ(ニューロン数)を返す
		unsigned int LayerSize(unsigned int LayerNumber) const
		{/*
		 LayerNumber層目のニューロンの数を返す

		 LayerNumber はパーセプトロンの層数を最大値，最小値を 1 とする．
		 */
			return m_NetworkForm[LayerNumber - 1];
		}

	public:
		// コンストラクタ
		perceptron_parameter(std::vector<unsigned int> NetworkForm)
		{
			m_NetworkForm = NetworkForm;

			unsigned int Size = NetworkSize();

			m_NeuronOutput.resize(Size);			// 0 <= size() <= NetworkSize()
			m_NeuronValue.resize(Size - 1);			// 0 <= size() <= NetworkSize() - 1
			m_Threshold.resize(Size - 1);	// 0 <= size() <= NetworkSize() - 1
			m_Weight.resize(Size - 1);		// 0 <= size() <= NetworkSize() - 1

			m_NeuronOutput[0].resize(LayerSize(1));							// 1層目のニューロンの数に合わせて調整
			for (unsigned int i = 0; i < Size - 1; ++i)
			{
				m_NeuronOutput[i + 1].resize(LayerSize(i + 2));				// i+2層目のニューロンの数に合わせて調整
				m_NeuronValue[i].resize(LayerSize(i + 2));					// i+2層目のニューロンの数に合わせて調整
				m_Threshold[i].resize(LayerSize(i + 2));					// i+2層目のニューロンの数に合わせて調整
				m_Weight[i].resize(LayerSize(i + 1) * LayerSize(i + 2));	// i+1層目とi+2層目のニューロンの数に合わせて調整
			}
		}

	};
}

#endif
