#include "metrics.h"

#include <cmath>

// взвешенное евклидово расстояние в квадрате
// используем квадрат там где нужно только сравнение — sqrt не нужен
double weighted_sq_distance(const FeatureVec& a, const FeatureVec& b,
                            const Weights& w) {
    double s = 0.0;
    for (int i = 0; i < FEATURE_DIM; ++i) {
        double d = a[i] - b[i];
        s += w[i] * d * d;
    }
    return s;
}

double weighted_distance(const FeatureVec& a, const FeatureVec& b,
                         const Weights& w) {
    return std::sqrt(weighted_sq_distance(a, b, w));
}
