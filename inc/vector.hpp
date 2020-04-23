#ifndef IG_VECTOR_HPP
#define IG_VECTOR_HPP

#include <vector>
#include <initializer_list>

namespace hoge {
    template <class T>
    class tensor {
        std::vector<std::size_t> dim; // 各添字の次元
        std::size_t rank;             // dim の size
        std::size_t size;             // dim[0] * ... * 1
        std::size_t order;            // dim[0] + ... + 0
        std::vector<T> v;

        // 添字変換
        template <class I, class... Is>
        std::size_t _getIndex(std::size_t r, I i, Is... is) const { return i + dim[r] * _getIndex<Is...>(r+1, is...); }
        template <class I, class... Is>
        std::size_t _getIndex(std::size_t r, I i) const { return i; }
        template <class... Is>
        std::size_t getIndex(Is... is) const { return _getIndex(0, is...); }

    public:
        template <class U>
        tensor(std::initializer_list<U> il) {
            dim.resize(il.size());
            int i = 0;
            for (auto itr = il.begin(); itr != il.end(); itr++) dim[i] = *itr;

            rank = dim.size();
            size = 1;
            order = 0;
            for (int i = 0; i < rank; i++) { size *= dim[i]; order += dim[i]; }

            v.resize(size);
        }
        // アクセス
        template <class... Is>
              T &operator () (Is... is)         { return v[getIndex<Is...>(is...)]; }
        template <class... Is>
        const T &operator () (Is... is) const   { return v[getIndex<Is...>(is...)]; }
        template <class... Is>
              T  operator () (Is... is) const&& { return v[getIndex<Is...>(is...)]; }
    };
}

#endif

