#include <iostream>
#include "step.hpp"

int main() {
    constexpr int n = 3;
    const float d[n*n] = {
        0, 8, 2,
        1, 0, 9,
        4, 5, 0,
    };
    float r[n*n];
    step(r, d, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << r[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}
