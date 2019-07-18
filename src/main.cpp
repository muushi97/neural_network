#include <iostream>

#include "dual_number.hpp"
#include "functions.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "tensor.hpp"

using namespace std;

int main() {
    tensor<double, 4, 3> a = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12} };

    cout << a(0, 2) << endl;

    {
        auto ind = a.begin();

        for (int i = 0; i < a.size(); i++) {
            cout << ind[0] << ", " << ind[1] << " : " << a(ind) << endl;
            ind.next();
        }
    }

    {
        indices<4> ind;
        indices<3> jnd;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                //cout << ind[0] << ", " << jnd[1] << " : " << a(ind, jnd) << endl;
                jnd.next();
            }
            ind.next();
        }
    }

    return 0;
}

