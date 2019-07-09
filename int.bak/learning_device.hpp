#ifndef IG_LEARNING_DEVICE_HPP
#define IG_LEARNING_DEVICE_HPP

#include <vector>

class perceptron;
class perceptron_parameter;

class learning_device
{
private:
    double m_LearningCoefficient;

public:
    // コンストラクタ
    learning_device(double LearningCoefficient) {
        setLearningCoefficient(LearningCoefficient);
    }

    // 学修係数のセット
    void setLearningCoefficient(double LearningCoefficient) {
        m_LearningCoefficient = LearningCoefficient;
    }

    // 重み，及び閾値の更新料を計算
    void calculate_difference(perceptron &network, perceptron_parameter &parameter, std::vector<double> InputSignal, std::vector<double> TeacherSignal) {
        network.input(InputSignal);
        network.propagate();

        double ganma;

        unsigned int NetworkSize = network.NetworkSize();

        // 閾値更新
        for (unsigned int j = network.LayerSize(NetworkSize); j > 0; --j) {
            ganma = network.NeuronOutput(NetworkSize, j) - TeacherSignal[j - 1];
            parameter.Threshold(NetworkSize, j) = ganma * network.m_ActivationFunction->activate_differential(network.NeuronValue(NetworkSize, j));
        }
        for (unsigned int i = NetworkSize - 1; i > 1; --i) {
            for (unsigned int j = network.LayerSize(i); j > 0; --j) {
                ganma = 0.0;
                for (unsigned int k = network.LayerSize(i + 1); k > 0; --k) {
                    ganma += parameter.Threshold(i + 1, k) * network.Weight(i, j, k);
                }

                parameter.Threshold(i, j)
                    = ganma * network.m_ActivationFunction->activate_differential(network.NeuronValue(i, j));
            }
        }

        // 重み更新
        for (unsigned int i = NetworkSize - 1; i > 0; --i)
            for (unsigned int j = network.LayerSize(i + 1); j > 0; --j)
                for (unsigned int k = network.LayerSize(i); k > 0; --k)
                    parameter.Weight(i, k, j) = parameter.Threshold(i + 1, j) * network.NeuronOutput(i, k);
    }

    // 更新量分，更新する
    void learn(perceptron &network, perceptron_parameter &parameter);

};

#endif

