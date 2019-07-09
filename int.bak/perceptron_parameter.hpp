#ifndef IG_PERCEPTRON_PARAMETER_HPP
#define IG_PERCEPTRON_PARAMETER_HPP

#include <vector>
#include <valarray>

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
    double &NeuronOutput(unsigned int LayerNumber, unsigned int NeuronNumber);
    // あるニューロンの値を参照する
    double &NeuronValue(unsigned int LayerNumber, unsigned int NeuronNumber);
    // あるニューロンの閾値を参照する
    double &Threshold(unsigned int LayerNumber, unsigned int NeuronNumber);
    // あるニューロンと，あるニューロン間の重みを参照する
    double &Weight(unsigned int LayerNumber, unsigned int SourceNeuronNumber, unsigned int DistinationNeuronNumber);

    // ネットワークのサイズ(層数)を返す
    unsigned int NetworkSize() const;
    // ある層のサイズ(ニューロン数)を返す
    unsigned int LayerSize(unsigned int LayerNumber) const;

public:
    // コンストラクタ
    perceptron_parameter(std::vector<unsigned int> NetworkForm);

};

#endif

