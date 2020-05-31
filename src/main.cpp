#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <memory>
#include <random>
#include <iomanip>
#include <sstream>

#include "trainer.hpp"
#include "loader.hpp"

using namespace std;

template <class T>
std::size_t max_1rand_tensor(const tensor<T> &t) {
    std::size_t i = 0;
    for (int j = 1; j < t.dim(0); j++)
        if (t(i) < t(j))
            i = j;
    return i;
}

int main(int argc, char *argv[]) {

    return 0;
}

