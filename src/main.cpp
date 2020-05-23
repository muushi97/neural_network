#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>

// tensor
template <class T>
class tensor {
    std::vector<std::size_t> dim_; // 各添字の次元
    std::size_t size_;             // dim[0] * ... * 1
    std::size_t order_;            // dim[0] + ... + 0
    std::vector<T> v;

    // 添字変換
    template <class I, class... Is>
    std::size_t _getIndex(std::size_t r, I i, Is... is) const { return i + dim(r) * _getIndex<Is...>(r+1, is...); }
    template <class I, class... Is>
    std::size_t _getIndex(std::size_t r, I i) const { return i; }
    template <class... Is>
    std::size_t getIndex(Is... is) const { return _getIndex(0, is...); }

public:
    // 添字
    class indices {
        friend tensor<T>;

        std::vector<std::size_t> dim_;
        std::size_t size_;
        std::size_t order_;

        std::vector<std::size_t> ind;

        template <class I, class... Is>
        void ind_set(std::size_t r, I i, Is... is) {
            if (r < ind.size()) {
                ind[r] = i;
                ind_set(r+1, is...);
            }
        }

    public:
        template <class... Is>
        indices(const tensor<T> &par, Is... is) : ind(par.rank()), dim_(par.dim_), size_(par.size_), order_(par.order_) {
            if constexpr (sizeof...(Is) != 0) ind_set(0, is...);
            else for (int i = 0; i < ind.size(); i++) ind[i] = 0;
        }

        void next() {
            int i = 0;
            while (true) {
                ind[i]++;
                if (ind[i] < dim_[i]) break;

                ind[i] = 0;
                i++;
                if (i >= dim_.size()) break;
            }
        }
        void prev() {
            int i = 0;
            while (true) {
                ind[i]--;
                if (ind[i] >= 0) break;

                ind[i] = dim_[i] - 1;
                i++;
                if (i >= dim_.size()) break;
            }
        }
        std::size_t &index(std::size_t i) { return ind[i]; }
        std::size_t index(std::size_t i) const { return ind[i]; }
        std::size_t getIndex() const {
            std::size_t i = ind[rank()-1];
            for (int r = 1; r < rank(); r++)
                i = ind[rank()-1-r] + dim(rank()-1-r) * i;
            return i;
        }

        std::size_t dim(std::size_t i) const { return dim_[i]; }
        std::size_t rank() const { return dim_.size(); }
        std::size_t size() const { return size_; }
        std::size_t oder() const { return order_; }
    };

    tensor() { }
    template <class U>
    tensor(std::vector<U> il) {
        dim_.resize(il.size());
        int i = 0;
        for (auto itr = il.begin(); itr != il.end(); itr++) {
            dim_[i] = static_cast<std::size_t>(*itr);
            i++;
        }

        size_ = 1;
        order_ = 0;
        for (int i = 0; i < dim_.size(); i++) { size_ *= dim_[i]; order_ += dim_[i]; }

        v.resize(size_);
    }
    template <class U, std::enable_if_t<std::is_integral<U>::value, int> = 0>
    tensor(std::initializer_list<U> il) {
        dim_.resize(il.size());
        int i = 0;
        for (auto itr = il.begin(); itr != il.end(); itr++) {
            dim_[i] = static_cast<std::size_t>(*itr);
            i++;
        }

        size_ = 1;
        order_ = 0;
        for (int i = 0; i < dim_.size(); i++) { size_ *= dim_[i]; order_ += dim_[i]; }

        v.resize(size_);
    }
    // 添字生成
    indices begin() const { return indices(*this); }
    // アクセス
    template <class... Is>
          T &operator () (Is... is)            { return v[getIndex<Is...>(is...)]; }
    template <class... Is>
    const T &operator () (Is... is)    const   { return v[getIndex<Is...>(is...)]; }
    template <class... Is>
          T  operator () (Is... is)    const&& { return v[getIndex<Is...>(is...)]; }
          T &at(indices ind)         { return v[ind.getIndex()]; }
    const T &at(indices ind) const   { return v[ind.getIndex()]; }
          //T  at(indices ind) const&& { return v[ind.getIndex()]; }

    std::size_t dim(std::size_t i) const { return dim_[i]; }
    std::size_t rank() const { return dim_.size(); }
    std::size_t size() const { return size_; }
    std::size_t oder() const { return order_; }
};

// layer
template <class T>
class base_layer {
    std::vector<int> input_size;
    std::vector<int> output_size;

public:
    const int &inputSize(int i) const { return input_size[i]; }
    const std::vector<int> &inputSize() const { return input_size; }
    const int inputRank() const { return input_size.size(); }
    const int &outputSize(int i) const { return output_size[i]; }
    const std::vector<int> &outputSize() const { return output_size; }
    const int outputRank() const { return output_size.size(); }

    // コンストラクタ
    base_layer(std::vector<int> is, std::vector<int> os) : input_size(is), output_size(os) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const = 0;
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dx) const = 0;

    // パラメータへのアクセス
    virtual tensor<T> &parameter() = 0;
    virtual const tensor<T> &parameter() const = 0;

    // デストラクタ
    virtual ~base_layer() { }
};
// 全結合層
template <class T>
class fully_connected_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> weight;

public:
    // コンストラクタ
    fully_connected_layer(int is, int os) : base_layer<T>{{is}, {os}}, weight{is+1, os} { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int o = 0; o < base::outputSize(0); o++) {
            y(o) = weight(base::inputSize(0), o); // しきい値で初期化
            for (int i = 0; i < base::inputSize(0); i++) {
                y(o) += x(i) * weight(i, o);
            }
        }

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize(0) };
        tensor<T> dw{ base::inputSize(0) + 1, base::outputSize(0) };

        for (int i = 0; i < base::inputSize(0); i++) {
            dx(i) = 0;
            for (int o = 0; o < base::outputSize(0); o++) {
                dx(i) += dy(o) * weight(i, o);
                dw(i, o) = dy(o) * x(i);
            }
        }
        for (int o = 0; o < base::outputSize(0); o++)
            dw(base::inputSize(0), o) = dy(o);

        return { dx, dw };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return weight; }
    virtual const tensor<T> &parameter() const { return weight; }

    // デストラクタ
    virtual ~fully_connected_layer() { }
};
// sigmoid 関数
template <class T>
class sigmoid_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> alpha;
    T sig(T a, T x) const {
        //std::cout << "sigmoid : " << "x(" << x << ") -> sig(" << 1.0 / (1.0 + std::exp(-1.0 * a * x)) << ")" << std::endl;
        return 1.0 / (1.0 + std::exp(-1.0 * a * x));
    }

public:
    // コンストラクタ
    sigmoid_layer(int s) : base_layer<T>({s}, {s}), alpha({s}) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int i = 0; i < base::inputSize(0); i++) {
            y(i) = sig(1.0, x(i));
        }

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize() };
        tensor<T> da{ base::inputSize() };

        for (int i = 0; i < base::inputSize(0); i++) {
            T s = sig(1.0, x(i));
            //T s = sig(alpha(i), x(i));
            dx(i) = 1.0      * s * (1.0 - s) * dy(i);
            //dx(i) = alpha(i) * s * (1.0 - s);
            da(i) = x(i)     * s * (1.0 - s) * dy(i);
        }

        return { dx, da };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return alpha; }
    virtual const tensor<T> &parameter() const { return alpha; }

    // デストラクタ
    virtual ~sigmoid_layer() { }
};
// ReLU 関数
template <class T>
class ReLU_layer : public base_layer<T> {
    using base = base_layer<T>;
    tensor<T> alpha;

public:
    // コンストラクタ
    ReLU_layer(int s) : base_layer<T>({s}, {s}), alpha({s}) { }

    // 順伝播 : 前の層の入力を受け、次の層への入力を計算
    virtual tensor<T> propagate(const tensor<T> x) const {
        tensor<T> y{ base::outputSize() };// あと初期化

        for (int i = 0; i < base::inputSize(0); i++)
            y(i) = x(i) > 0.0 ? x(i) : 0.0;

        return y;
    }
    // 逆伝播 : 前の層の入力と次の層の誤差を受け、この層のニューロンの誤差と重みの誤差を計算
    virtual std::array<tensor<T>, 2> backpropagate(const tensor<T> x, const tensor<T> dy) const {
        tensor<T> dx{ base::inputSize() };

        for (int i = 0; i < base::inputSize(0); i++)
            dx(i) = x(i) > 0.0 ? 1.0 : -0.1;

        return { dx, tensor<T>{0} };
    }

    // パラメータへのアクセス
    virtual tensor<T> &parameter() { return alpha; }
    virtual const tensor<T> &parameter() const { return alpha; }

    // デストラクタ
    virtual ~ReLU_layer() { }
};

// レイヤーの別名
template <class T>
using layer = base_layer<T>;

template <class T>
class trainer;

// フィードフォワードネットワーク
template <class T>
class network {
    friend trainer<T>;

    std::vector<std::unique_ptr<layer<T>>> ls;

    // パラメータの取得
    tensor<T> &parameter(int i) {
        return ls[i]->parameter();
    }
    const tensor<T> &parameter(int i) const { return ls[i]->parameter(); }

    // 逆伝播
    std::array<tensor<T>, 2> backpropagate(int i, const tensor<T> x, const tensor<T> dy) { return ls[i]->backpropagate(x, dy); }

public:
    // コンストラクタ
    network(std::initializer_list<layer<T>*> il) {
        for (auto itr = il.begin(); itr != il.end(); itr++)
            ls.emplace_back(*itr);
    }

    // 順伝播
    tensor<T> propagate(int i, const tensor<T> x) { return ls[i]->propagate(x); }
    tensor<T> propagate(const tensor<T> x) {
        tensor<T> y = x;
        for (int i = 0; i < ls.size(); i++) {
            y = propagate(i, y);
        }
        return y;
    }

    std::size_t layer_size() const { return ls.size(); }
};

// 学習機
template <class T>
class trainer {
public:
    // mean squared error
    double mse(const tensor<T> x, const tensor<T> t) {
        T e = 0;
        T c = 0;
        T y, u;
        auto ind = x.begin();
        for (auto i = 0; i < x.size(); i++) {
            y = std::pow(x.at(ind) - t.at(ind), 2) - c;
            u = e + y;
            c = (u - e) - y;
            e = u;
            ind.next();
        }
        return e;
    }
private:
    tensor<T> d_mse(const tensor<T> x, const tensor<T> t) {
        tensor<T> dx = x;
        auto ind = x.begin();
        for (int i = 0; i < x.size(); i++) {
            dx.at(ind) -= t.at(ind);
            ind.next();
        }
        return dx;
    }

public:
    T error;
    T previous_error() const { return error; }

    void set_parameter_by_uniform_distribution(network<T> &net, T a, T b) {
        std::random_device seed_gen;
        std::mt19937_64 engine(seed_gen());

        std::uniform_real_distribution<> dist(a, b);

        for (int l = 0; l < net.layer_size(); l++) {
            //int l_ = net.layer_size() - 1 - l;
            int l_ = l;
            auto ind = net.parameter(l_).begin();
            for (int i = 0; i < net.parameter(l_).size(); i++) {
                net.parameter(l_).at(ind) = dist(engine);
                ind.next();
            }
        }
    }

    std::vector<tensor<T>> backpropagate(network<T> &net, const tensor<T> x, const tensor<T> t) {
        int N = net.layer_size();
        std::vector<tensor<T>> xs(N + 1);
        std::vector<tensor<T>> ws(N);

        // propagation
        xs[0] = x;
        for (int i = 0; i < N; i++) {
            xs[i + 1] = net.propagate(i, xs[i]);
        }

        // error
        error = mse(xs[N], t);
        tensor<T> dy = d_mse(xs[N], t);

        // backpropagation
        std::array<tensor<T>, 2> res;
        for (int i = 0; i < N; i++) {
            res = net.backpropagate(N-1-i, xs[N-1-i], dy);
            dy = res[0];
            ws[N-1-i] = res[1];
        }

        return ws;
    }
    void parameter_update(network<T> &net, double eta, const std::vector<tensor<T>> xs, const std::vector<tensor<T>> ts) {
        std::vector<tensor<T>> ws = backpropagate(net, xs[0], ts[0]);
        std::vector<tensor<T>> ws_;
        for (int i = 1; i < xs.size(); i++) {
            ws_ = backpropagate(net, xs[i], ts[i]);
            for (int l = 0; l < ws_.size(); l++) {
                auto ind = ws[l].begin();
                for (int j = 0; j < ws[l].size(); j++) {
                    ws[l].at(ind) += ws_[l].at(ind);
                    ind.next();
                }
            }
        }

        for (int l = 0; l < ws.size(); l++) {
            auto ind = ws[l].begin();
            for (int j = 0; j < ws[l].size(); j++) {
                net.parameter(l).at(ind) = net.parameter(l).at(ind) - eta * ws[l].at(ind);
                ind.next();
            }
        }
    }
};

using namespace std;

int main() {
    network<double>    n{ new fully_connected_layer<double>(2, 4)
                        , new sigmoid_layer<double>(4)
                        , new fully_connected_layer<double>(4, 1)
                        , new sigmoid_layer<double>(1)
                        };

    tensor<double> temp1{2}, temp2{1};

    std::vector<tensor<double>> xs, ts;
    temp1(0) = 0.0; temp1(1) = 0.0;   temp2(0) = 0.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 0.0; temp1(1) = 1.0;   temp2(0) = 1.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 1.0; temp1(1) = 0.0;   temp2(0) = 1.0;
    xs.push_back(temp1); ts.push_back(temp2);
    temp1(0) = 1.0; temp1(1) = 1.0;   temp2(0) = 0.0;
    xs.push_back(temp1); ts.push_back(temp2);

    trainer<double> tr;

    tr.set_parameter_by_uniform_distribution(n, -0.4, 0.4);
    for (int i = 0; i < 100000; i++) {
        tr.parameter_update(n, 0.1, xs, ts);
        //getchar();
    }

    tensor<double> y;
    y = n.propagate(xs[0]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[1]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[2]);
    std::cout << y(0) << std::endl;
    y = n.propagate(xs[3]);
    std::cout << y(0) << std::endl;

    return 0;
}

