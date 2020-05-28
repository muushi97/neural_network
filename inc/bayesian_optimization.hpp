#ifndef IG_BAYESIAN_OPTIMIZATION_HPP
#define IG_BAYESIAN_OPTIMIZATION_HPP

#include "gaussian_process_regression.hpp"

class BO {
public:
    enum acquisition_function_type { pi  // Probability of Improvement
                                   , ei  // Expected Improvement
                                   , lcb // GP Upper Confidence Bound
    };

private:
    GPR gpr;
    acquisition_function_type af_type;

    // acquisition function
    double acquisition_function() { return 0.0; }
    // maximize
    double newton() { return 0.0; }

public:
    BO(double sig2, acquisition_function_type af) : gpr(sig2), af_type(af) {}

    // 教師データを追加
    void add_train_data(const tensor<double> &x, double y_) {
        gpr.add_train_data(x, y_);
    }
    void add_train_data(const std::vector<tensor<double>> &x, std::vector<double> &y_) {
        gpr.add_train_data(x, y_);
    }

    // 最大値の探索

};

#endif

