#ifndef IG_GAUSSIAN_PROCESS_REGRESSION_HPP
#define IG_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <vector>

#include "tensor.hpp"

class GPR {
public:
    enum kernel_type { se   // squared exponential kernel
                     , m52  // ARD Matern 5/2 kernel
    };

private:
    double sigma2;                 // ノイズの標準偏差
    std::vector<tensor<double>> X;  // 教師データの入力(行列)
    std::vector<double> y;          // 教師データの出力(ベクトル)
    kernel_type type;               // カーネル
    tensor<double> K;               // カーネルと教師データから得られる行列に標準偏差を足した行列

    double kernel(const tensor<double> &x1, const tensor<double> &x2) {
        if (type == se) {
            double s = 0.0;
            for (int i = 0; i < x1.dim(0); i++)
                s += std::pow(x1(i) - x2(i), 2);
            return 1.0 * std::exp(-0.5 * s);
        }
        else if (type == m52) {
            double s = 0.0;
            for (int i = 0; i < x1.dim(0); i++)
                s += std::pow(x1(i) - x2(i), 2);
            double s_ = 5.0 * std::sqrt(s);
            //std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : " << 1.0 * (1.0 + s_ + 5.0 / 3.0 * s) * std::exp(-1.0 * s) << std::endl;
            return 1.0 * (1.0 + s_ + 5.0 / 3.0 * s) * std::exp(-1.0 * s_);
        }
        else return 0.0;
    }

    void calc_K() {
        K = tensor<double>{ X.size(), X.size() };
        double sig2 = sigma2;

        for (int i = 0; i < X.size(); i++)
            for (int j = 0; j < X.size(); j++)
                K(i, j) = kernel(X[i], X[j]) + (i == j ? sig2 : 0.0);
    }
    std::vector<double> calc_k_ast_vec(const tensor<double> &x_ast) {
        std::vector<double> k_ast(X.size());
        for (int i = 0; i < X.size(); i++)
            k_ast[i] = kernel(x_ast, X[i]);
        return k_ast;
    }

    // x A = b を解く  <=>  x = b A^-1 を計算する
    std::vector<double> mul_vec_inv(std::vector<double> b, tensor<double> A) {
        int n = b.size();
        for (int i = 0; i < n; i++) {
            int pivot = i;
            for (int j = i; j < n; j++)
                if (std::abs(A(i, j)) > std::abs(A(i, pivot))) pivot = j;

            if (std::abs(A(i, pivot)) < 1e-8) return std::vector<double>();

            double temp;
            for (int j = i; j < n; j++) {
                temp = A(j, i);
                if (j == i) A(j, i) = A(j, pivot);
                else        A(j, i) = A(j, pivot) / A(i, i);
                if (i != pivot)
                    A(j, pivot) = temp;
            }
            temp = b[i];
            b[i] = b[pivot] / A(i, i);
            if (i != pivot)
                b[pivot] = temp;

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    for (int k = i + 1; k < n; k++)
                        A(k, j) -= A(i, j) * A(k, i);
                    b[j] -= A(i, j) * b[i];
                }
            }
            //std::cout << "A" << std::endl;
            //for (int i = 0; i < b.size(); i++)
            //    for (int j = 0; j < b.size(); j++)
            //        std::cout << A(i, j) << ",\n"[j==b.size()-1];
            //std::cout << "b" << std::endl;
            //for (int i = 0; i < b.size(); i++)
            //    std::cout << b[i] << ",\n"[i==b.size()-1];
            //std::cout << std::endl;
        }
        return b;
    }

public:
    void jordan_test() {
        tensor<double> A{3, 3};
        std::vector<double> b(3);

        A(0, 0) = + 8.0; A(0, 1) = + 2.0; A(0, 2) = +10.0;
        A(1, 0) = + 5.0; A(1, 1) = - 3.0; A(1, 2) = + 2.0;
        A(2, 0) = - 6.0; A(2, 1) = + 2.0; A(2, 2) = + 3.0;
        b[0]    = - 6.0; b[1]    = + 4.0; b[2]    = +26.0;

        std::vector<double> x = mul_vec_inv(b, A);

        for (int i = 0; i < x.size(); i++)
            std::cout << x[i] << ",\n"[i==x.size()-1];
    }

    GPR(double sig2, kernel_type t) : sigma2(sig2), type(t) { }

    // 教師データを追加
    void add_train_data(const tensor<double> &x, double y_) {
        X.push_back(x);
        y.push_back(y_);
        calc_K();
    }
    void add_train_data(const std::vector<tensor<double>> &x, std::vector<double> &y_) {
        for (int i = 0; i < x.size(); i++) {
            X.push_back(x[i]);
            y.push_back(y_[i]);
        }
        calc_K();
    }

    // 期待値 e と分散 v を推定
    std::array<double, 2> EV(const tensor<double> &x) {
        std::vector<double> k_ast = calc_k_ast_vec(x);

        std::vector<double> temp = mul_vec_inv(k_ast, K);

        double e = 0.0;
        double v = kernel(x, x);
        //std::cout << "aaaaaaaaa : " << e << ", " << v << std::endl;
        for (int i = 0; i < X.size(); i++) {
            e += temp[i] * y[i];
            v -= temp[i] * k_ast[i];
            //std::cout << "  bbbbbbbbb : " << temp[i] << ", " << y[i] << ", " << k_ast[i] << " ==>> " << e << ", " << v << std::endl;
        }
        //std::cout << "  ccccccccc : " << e << ", " << v << std::endl;
        return { e, v };
    }
};


#endif

