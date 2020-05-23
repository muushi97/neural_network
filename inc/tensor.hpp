#ifndef IG_TENSOR_HPP
#define IG_TENSOR_HPP

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

#endif
