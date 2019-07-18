#ifndef IG_NETWORK_HPP
#define IG_NETWORK_HPP

#include <tuple>

template <typename T, typename U>
struct marge_tuple_parameter;
template <typename... Ts, typename... Us>
struct marge_tuple_parameter<std::tuple<Ts...>, std::tuple<Us...>> { using type = std::tuple<Ts..., Us...>; };
template <typename... Ts>
struct parameter_splitter;
template <typename T1, typename T2, typename... Ts>
struct parameter_splitter<T1, T2, Ts...> {
    using odd = typename marge_tuple_parameter<std::tuple<T1>, typename parameter_splitter<Ts...>::odd>::type;
    using even = typename marge_tuple_parameter<std::tuple<T2>, typename parameter_splitter<Ts...>::even>::type;
    static constexpr std::size_t size = parameter_splitter<Ts...>::size + 1; };
template <typename T1, typename T2, typename T3>
struct parameter_splitter<T1, T2, T3> {
    using odd = std::tuple<T1, T3>;
    using even = std::tuple<T2>;
    static constexpr std::size_t size = 2; };

template <class... Ts>
class network {
private:
    typename parameter_splitter<Ts...>::odd layers;
    typename parameter_splitter<Ts...>::even linkings;
    static constexpr std::size_t layer_number = parameter_splitter<Ts...>::size;

    template <std::size_t I>
    void propagate() {
        if (I >= layer_number - 1) return;
        std::get<I+1>(layers).propagate(std::get<I>(linkings), std::get<I>(layers));
        propagate<I+1>();
    }

public:
    network() { }

    template <class T, std::size_t... Ns>
    void propagate(const tensor<T, Ns...> &x) {
        std::get<0>(layers).set(x);
        propagate<0>();
    }
};

#endif

