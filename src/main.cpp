#include <cstdio>
#include <iostream>

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
    static inline auto mul(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x * b.x)>(a.x * b.x, a.x_dash * b.x + a.x * b.x_dash); }
    static inline auto div(const dual_number<U1> &a, const dual_number<U2> &b) {
        return dual_number<decltype(a.x / b.x)>(a.x / b.x, (a.x_dash * b.x - a.x * b.x_dash) / (b.x * b.x)); }
};

using namespace std;

int main() {
    dual_number<double> a(1, 0);
    dual_number<double> b(10, 1);

    dual_number<double> c = a.add(a, b);
    cout << a.value() << endl;
    cout << b.value() << endl;
    cout << c.value() << endl;

    return 0;
}

