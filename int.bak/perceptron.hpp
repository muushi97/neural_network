#ifndef IG_PERCEPTRON_HPP
#define IG_PERCEPTRON_HPP

#include <vector>

#include "perceptron_parameter.hpp"

#include "_activation_function.hpp"

class perceptron : public perceptron_parameter
{
    friend class initializer;
    friend class learning_device;

private:
    std::vector<double> m_InputSignal;
    std::vector<double> m_OutputSignal;

    _activation_function *m_ActivationFunction;

public:
    // コンストラクタ
    perceptron(std::vector<unsigned int> NetworkForm, _activation_function *ActivationFunction);

    // 入力
    void input(std::vector<double> InputSignal);
    // 順伝播
    void propagate();
    // 出力
    const std::vector<double> &output() const;

    // 入力，順伝播，出力をまとめて行う
    const std::vector<double> &operator() (std::vector<double> InputSignal);

    // ネットワークのサイズ(層数)を返す
    unsigned int NetworkSize() const;
    // ある層のサイズ(ニューロン数)を返す
    unsigned int LayerSize(unsigned int LayerNumber) const;

};

#endif

