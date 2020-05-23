#ifndef IG_TENSOR_HPP
#define IG_TENSOR_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>


namespace {
    class shape {
        std::vector<std::size_t> dim_;
        std::size_t size_;
        std::size_t order_;

        void recalc() {
            size_ = 1;
            order_ = 0;
            for (int i = 0; i < dim_.size(); i++) {
                size_ *= dim_[i];
                order_ += dim_[i];
            }
        }

    public:
        shape() : shape({}) { }
        shape(std::initializer_list<std::size_t> il) : dim_{il} {
            recalc();
        }

        void reshape_rank(std::size_t r) {
            std::vector<std::size_t> d(r);
            if (rank() < r) {
                for (int i = 0; i < rank(); i++)
                    d[i] = dim_[i];
                for (int i = rank(); i < r; i++)
                    d[i] = 1;
            }
            else if (rank() > r) {
                for (int i = 0; i < r; i++)
                    d[i] = dim_[i];
            }
            dim_ = std::move(d);
            recalc();
        }
        void set_dim(std::size_t i, std::size_t d) {
            dim_[i] = d;
            recalc();
        }

        std::size_t dim(std::size_t i) const { return dim_[i]; }
        std::size_t rank() const { return dim_.size(); }
        std::size_t size() const { return size_; }
        std::size_t order() const { return order_; }

        friend shape merge_shape(const shape &s1, const shape &s2);
        friend shape merge_shape(std::size_t i1, const shape &s1);
        friend shape merge_shape(const shape &s1, std::size_t i1);
    };

    shape merge_shape(const shape &s1, const shape &s2) {
        shape s;
        s.dim_.resize(s1.rank() + s2.rank());
        s.size_ = s1.size() * s2.size();
        s.order_ = s1.order() + s2.order();
        for (int i = 0; i < s1.rank(); i++)
            s.dim_[i            ] = s1.dim(i);
        for (int i = 0; i < s2.rank(); i++)
            s.dim_[i + s1.rank()] = s2.dim(i);
        return s;
    }
    shape merge_shape(std::size_t i1, const shape &s1) {
        shape s;
        s.dim_.resize(1 + s1.rank());
        s.size_ = i1 * s1.size();
        s.order_ = i1 + s1.order();
        for (int i = 0; i < 1; i++)
            s.dim_[i            ] = i1;
        for (int i = 0; i < s1.rank(); i++)
            s.dim_[i + 1        ] = s1.dim(i);
        return s;
    }
    shape merge_shape(const shape &s1, std::size_t i1) {
        shape s;
        s.dim_.resize(s1.rank() + 1);
        s.size_ = s1.size() * i1;
        s.order_ = s1.order() + i1;
        for (int i = 0; i < s1.rank(); i++)
            s.dim_[i            ] = s1.dim(i);
        for (int i = 0; i < 1; i++)
            s.dim_[i + s1.rank()] = i1;
        return s;
    }
}

class indices {
    shape s;

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
    indices(const shape &sh, Is... is) : s(sh), ind(sh.rank()) {
        for (int i = 0; i < ind.size(); i++) ind[i] = 0;
        if constexpr (sizeof...(Is) != 0) ind_set(0, is...);
    }

    bool next() {
        int i = 0;
        while (true) {
            ind[i]++;
            if (i == s.rank()-1) {
                if (ind[i] <= s.dim(i)) return true;
            }
            else 
                if (ind[i] < s.dim(i)) return true;

            ind[i] = 0;
            i++;
            if (i >= s.rank()) return false;
        }
    }
    bool prev() {
        int i = 0;
        while (true) {
            if (ind[i] > 0) {
                ind[i]--;
                return true;
            }

            if (i == s.rank()-1) {
                ind[i] = s.dim(i);
                for (i = 0; i < s.rank()-1; i++)
                    ind[i] = 0;
                return false;
            }
            else  {
                ind[i] = s.dim(i) - 1;
                i++;
            }
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

    std::size_t dim(std::size_t i) const { return s.dim(i); }
    std::size_t rank() const { return s.rank(); }
    std::size_t size() const { return s.size(); }
    std::size_t oder() const { return s.order(); }
};

bool operator == (const indices &i1, const indices &i2) {
    if (i1.rank() != i2.rank()) return false;
    for (int i = 0; i < i1.rank(); i++)
        if (i1.index(i) != i2.index(i)) return false;
    return true;
}
bool operator != (const indices &i1, const indices &i2) {
    return i1 == i2 ? false : true;
}


// tensor
template <class T>
class tensor {
    shape s;
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
    tensor() : tensor({}) { }
    tensor(std::vector<std::size_t> iv) : s() {
        s.reshape_rank(iv.size());
        for (int i = 0; i < iv.size(); i++)
            s.set_dim(i, iv[i]);
        v.resize(s.size());
    }
    tensor(std::initializer_list<std::size_t> il) : s(il) {
        v.resize(s.size());
    }
    // 添字生成
    indices begin() const { return indices(s); }
    indices end() const { indices i(s); i.prev(); return i; }
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

    std::size_t dim(std::size_t i) const { return s.dim(i); }
    std::size_t rank() const { return s.rank(); }
    std::size_t size() const { return s.size(); }
    std::size_t oder() const { return s.order(); }
};

#endif
