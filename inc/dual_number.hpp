#ifndef IG_DUAL_NUMBER_HPP
#define IG_DUAL_NUMBER_HPP

#include <cmath>

// 二重数
template <class T>
class dual_number {
private:
    T x;
    T x_dash;

public:
    dual_number() : dual_number(0, 0) { }
    dual_number(const T x, const T x_dash) : x(x), x_dash(x_dash) { }

    T &value() { return x; }
    T &diff() { return x_dash; }
    const T &value() const { return x; }
    const T &diff() const { return x_dash; }

    template <class U1, class U2>
    static inline auto add(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x + b.x)>(a.x + b.x, a.x_dash + b.x_dash); }
    template <class U1, class U2>
    static inline auto sub(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x - b.x)>(a.x - b.x, a.x_dash - b.x_dash); }
    template <class U1, class U2>
    static inline auto mul(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x * b.x)>(a.x * b.x, a.x_dash * b.x + a.x * b.x_dash); }
    template <class U1, class U2>
    static inline auto div(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x / b.x)>(a.x / b.x, (a.x_dash * b.x - a.x * b.x_dash) / (b.x * b.x)); }
};
template <class U1, class U2>
auto operator + (const dual_number<U1> &a, const dual_number<U2> &b) { return a.add(a, b); }
template <class U1, class U2>
auto operator - (const dual_number<U1> &a, const dual_number<U2> &b) { return a.sub(a, b); }
template <class U1, class U2>
auto operator * (const dual_number<U1> &a, const dual_number<U2> &b) { return a.mul(a, b); }
template <class U1, class U2>
auto operator / (const dual_number<U1> &a, const dual_number<U2> &b) { return a.div(a, b); }
template <class U>
auto pow(const dual_number<U> &a, double i) {
    if (i == 0.0) return dual_number<U>(1, 0);
    else return dual_number<decltype(std::pow(a.value(), i))>(std::pow(a.value(), i), a.diff() * i * std::pow(a.value(), i-1)); }
template <class U>
auto sin(const dual_number<U> &a) {
    return dual_number<decltype(std::sin(a.value()))>(std::sin(a.value()), a.diff() * std::cos(a.value())); }
template <class U>
auto cos(const dual_number<U> &a) {
    return dual_number<decltype(std::cos(a.value()))>(std::cos(a.value()), -a.diff() * std::sin(a.value())); }
template <class U>
auto tan(const dual_number<U> &a) {
    return sin(a) / cos(a); }
template <class U>
auto exp(const dual_number<U> &a) {
    return dual_number<decltype(std::exp(a.value()))>(std::exp(a.value()), a.diff() * std::exp(a.value())); }
template <class U>
auto log(const dual_number<U> &a) {
    return dual_number<decltype(std::log(a.value()))>(std::log(a.value()), a.diff() / a.value()); }

#endif
