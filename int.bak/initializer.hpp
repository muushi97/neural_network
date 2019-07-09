#ifndef IG_INITIALIZER_HPP
#define IG_INITIALIZER_HPP

#include <random>

class perceptron;

class initializer
{
private:
    std::mt19937 m_engine;
    double m_min, m_max;

public:
    // 今ストラクた
    initializer() : initializer(-1.0, 1.0) { }
    initializer(double min, double max) : m_engine(std::time(NULL)) {
        m_min = min; m_max = max;
    }

    // 初期化
    void initialize(perceptron &Network) {
        // (最小値, 最大値)
        std::uniform_real_distribution<> ur_dist(m_min, m_max);

        // 閾値の初期化
        for (unsigned int i = 2; i <= Network.NetworkSize(); ++i)
            for (unsigned int j = 1; j <= Network.LayerSize(i); ++j)
                Network.Threshold(i, j) = ur_dist(m_engine);

        // 重みの初期化
        for (unsigned int i = 1; i <= Network.NetworkSize() - 1; ++i)
            for (unsigned int j = 1; j <= Network.LayerSize(i); ++j)
                for (unsigned int k = 1; k <= Network.LayerSize(i + 1); ++k)
                    Network.Weight(i, j, k) = ur_dist(m_engine);
    }
};

#endif

