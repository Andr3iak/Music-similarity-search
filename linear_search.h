#ifndef LINEAR_SEARCH_H
#define LINEAR_SEARCH_H

#include <utility>
#include <vector>

#include "features.h"
#include "metrics.h"

// Возвращает k ближайших к query индексов из points (отсортированных
// по возрастанию расстояния). second — квадрат взвешенного расстояния.
std::vector<std::pair<double, int>> linear_knn(
    const std::vector<FeatureVec>& points, const FeatureVec& query,
    const Weights& w, int k);

#endif
