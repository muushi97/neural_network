#include <cstdio>
#include <iostream>
#include <array>
#include <tuple>

template <std::size_t... Ns>
struct mul_template_parameter;
template <std::size_t N, std::size_t... Ns>
struct mul_template_parameter<N, Ns...> { static constexpr std::size_t value = N * mul_template_parameter<Ns...>::value; };
template <>
struct mul_template_parameter<> { static constexpr std::size_t value = 1; };

template <std::size_t I, std::size_t J, std::size_t... Ns>
struct at_template_parameter;
template <std::size_t I, std::size_t J, std::size_t N, std::size_t... Ns>
struct at_template_parameter<I, J, N, Ns...> {
    static constexpr std::size_t value = at_template_parameter<I, J+1, Ns...>::value;
};
template <std::size_t I, std::size_t N, std::size_t... Ns>
struct at_template_parameter<I, I, N, Ns...> {
    static constexpr std::size_t value = N;
};

template <class T, std::size_t... Ns>
class tensor {
    static constexpr std::size_t Rank = sizeof...(Ns);
    static constexpr std::size_t Dimension[sizeof...(Ns)] = { Ns... };
    std::array<T, mul_template_parameter<Ns...>::value> t;


public:
    tensor() {
    }

    constexpr std::size_t dimension(int i) const { return i < Rank ? Dimension[i] : 0; }
    constexpr std::size_t rank() const { return Rank; }
    constexpr std::size_t order() const {
        int o = 0;
        for (int i = 0; i < rank(); i++) o += dimension(i);
        return o;
    }
    constexpr std::size_t size() const { return t.size(); }

};
template <class T, std::size_t... Ns>
constexpr std::size_t tensor<T, Ns...>::Dimension[];

using namespace std;

int main() {
    //vector<double> w;
    tensor<double> a;

    cout << a.dimension(0) << endl;
    cout << a.dimension(1) << endl;
    cout << a.dimension(2) << endl;
    cout << a.dimension(3) << endl;
    cout << a.rank() << endl;
    cout << a.order() << endl;
    cout << a.size() << endl;

    return 0;
}

