#ifndef IG_TENSOR_HPP
#define IG_TENSOR_HPP

#include <array>
#include <initializer_list>

template <typename T, template <typename...> class U, std::size_t N> 
struct type_nesting { using type = U<typename type_nesting<T, U, N-1>::type>; };
template <typename T, template <typename...> class U>
struct type_nesting<T, U, 0> { using type = T; };

template <template <std::size_t... Ms> typename T, std::size_t S, std::size_t... Ns>
struct split_indices;

// 添字クラス
template <std::size_t... Ns>
class indices {
    static inline constexpr std::size_t Rank = sizeof...(Ns);
    static inline constexpr std::size_t Dim[sizeof...(Ns)] = { Ns... };
    static inline constexpr std::size_t Order = (Ns + ... + 0);
    static inline constexpr std::size_t Size = (Ns * ... * 1);

    std::array<std::size_t, sizeof...(Ns)> ind;

public:
    // コンストラクタ
    indices() { for (int i = 0; i < Rank; i++) ind[i] = 0; }
    template <class... T>
    indices(T... Ms) : ind{Ms...} { }
    indices(std::initializer_list<std::size_t> x) : ind(x) { }
    indices(const indices<Ns...> &x) { for (std::size_t i = 0; i < Rank; i++) ind[i] = x.ind[i]; }

    // 代入
    const indices<Ns...> &operator=(std::initializer_list<std::size_t> x) { ind = x; return *this; }
    const indices<Ns...> &operator=(const indices<Ns...> &x) {
        for (std::size_t i = 0; i < Rank; i++) ind[i] = x.ind[i]; return *this; }

    bool next() {
        for (int i = Rank - 1; i >= 0; i--) {
            ind[i]++;
            if (ind[i] >= Dim[i]) ind[i] = 0;
            else return true;
        }
        return false;
    }

    std::size_t &operator [] (std::size_t i) { return ind[i]; }
    const std::size_t operator [] (std::size_t i) const { return ind[i]; }

};

template <class T, std::size_t... Ns>
class tensor;

template <typename T, typename... Us>
struct indices_checker;
template <typename T, std::size_t N, std::size_t... Ns, typename U, typename... Us>
struct indices_checker<tensor<T, N, Ns...>, U, Us...> {
    template <typename TEMP, std::enable_if_t<std::is_integral<U>::value, std::nullptr_t> = nullptr>
    static constexpr bool result(TEMP) {
        return indices_checker<tensor<T, Ns...>, Us...>::result(true);
    }
};

// テンソルクラス
template <class T, std::size_t... Ns>
class tensor {
    using type = T;
    using this_tensor = tensor<T, Ns...>;
    using i_type = std::size_t;
    static inline constexpr i_type Rank = sizeof...(Ns);
    static inline constexpr i_type Dim[sizeof...(Ns)] = { Ns... };
    static inline constexpr i_type Order = (Ns + ... + 0);
    static inline constexpr i_type Size = (Ns * ... * 1);

    std::array<T, Size> t;

    template <class I, class... Is>
    i_type getIndex(i_type r, I i, Is... is) const { return i + Dim[r] * getIndex<Is...>(r+1, is...); }
    template <class I, class... Is>
    i_type getIndex(i_type r, I i) const { return i; }

    template <class... Is>
    void init(T x, Is... is) { (*this)(is...) = x; }
    template <class U, class... Is>
    void init(U x, Is... is) { i_type i = 0; for (auto &&e : x) init(e, is..., i), i++; }

public:
    // コンストラクタ
    tensor() { }
    tensor(typename type_nesting<T, std::initializer_list, Rank>::type x) { init(x); }
    tensor(const tensor<T, Ns...> &x) { for (i_type i = 0; i < Size; i++) t[i] = x.t[i]; }

    // 代入
    const this_tensor &operator=(typename type_nesting<T, std::initializer_list, Rank>::type x) { init(x); return *this; }
    const this_tensor &operator=(const tensor<T, Ns...> &x) { for (i_type i = 0; i < Size; i++) t[i] = x.t[i]; return *this; }

    indices<Ns...> begin() const { return indices<Ns...>(); }

    // 次元や添字の情報等を得る関数
    constexpr i_type dimension(i_type i) const { return i < Rank ? Dim[i] : 0; }
    constexpr i_type rank() const { return Rank; }
    constexpr i_type order() const { return Order; }
    constexpr i_type size() const { return Size; }

    // 添字による要素へのアクセス用関数
    T &operator () (const indices<Ns...> &i) {
        i_type j = i[Rank - 1];
        for (int r = Rank - 2; r >= 0; r--) j = j * Dim[r] + i[r];
        return t[j];
    }
    const T &operator () (const indices<Ns...> &i) const {
        i_type j = i[Rank - 1];
        for (int r = Rank - 2; r > 0; r--) j = j * Dim[r] + i[r];
        return t[j];
    }
    template <i_type... M1s, i_type... M2s>
    T &operator () (const indices<M1s...> &i1, const indices<M1s...> &i2) {
        static_assert(Rank == sizeof...(M1s) + sizeof...(M2s), "");
        i_type j = i2[sizeof...(M2s) - 1];
        for (int r = sizeof...(M2s) - 2; r >= 0; r--) j = j * Dim[sizeof...(M1s) + r] + i2[r];
        for (int r = sizeof...(M1s) - 1; r >= 0; r--) j = j * Dim[r] + i2[r];
        return t[j];
    }
    template <i_type... M1s, i_type... M2s>
    const T &operator () (const indices<M1s...> &i1, const indices<M1s...> &i2) const {
        static_assert(Rank == sizeof...(M1s) + sizeof...(M2s), "");
        i_type j = i2[sizeof...(M2s) - 1];
        for (int r = sizeof...(M2s) - 2; r >= 0; r--) j = j * Dim[sizeof...(M1s) + r] + i2[r];
        for (int r = sizeof...(M1s) - 1; r >= 0; r--) j = j * Dim[r] + i2[r];
        return t[j];
    }
    template <class... Is, typename std::enable_if<sizeof...(Is) == sizeof...(Ns)>::type * = nullptr>
    T &operator () (Is... is) {
        return t[getIndex<Is...>(0, is...)];
    }
    template <class... Is, typename std::enable_if<sizeof...(Is) == sizeof...(Ns)>::type * = nullptr>
    const T &operator () (Is... is) const {
        return t[getIndex<Is...>(0, is...)];
    }
    template <class... Is, typename std::enable_if<sizeof...(Is) == sizeof...(Ns)>::type * = nullptr>
    T operator () (Is... is) const&& {
        return t[getIndex<Is...>(0, is...)];
    }

};

#endif
