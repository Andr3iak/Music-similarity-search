#ifndef METRICS_H
#define METRICS_H

#include <array>

#include "features.h"

using Weights = std::array<double, FEATURE_DIM>;

// Квадрат взвешенного евклидова расстояния (без sqrt — для k-d дерева).
double weighted_sq_distance(const FeatureVec& a, const FeatureVec& b,
                            const Weights& w);

// Взвешенное евклидово расстояние (с sqrt).
double weighted_distance(const FeatureVec& a, const FeatureVec& b,
                         const Weights& w);

#endif
