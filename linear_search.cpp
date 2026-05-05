#include "linear_search.h"

#include <algorithm>
#include <queue>

std::vector<std::pair<double, int>> linear_knn(
    const std::vector<FeatureVec>& points, const FeatureVec& query,
    const Weights& w, int k) {
    // max-куча на k элементов: храним k ближайших, на вершине — самый дальний
    // когда находим ближе — выталкиваем максимум и добавляем новый
    std::priority_queue<std::pair<double, int>> heap;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double d2 = weighted_sq_distance(points[i], query, w);
        if (static_cast<int>(heap.size()) < k) {
            heap.emplace(d2, i);
        } else if (d2 < heap.top().first) {
            heap.pop();
            heap.emplace(d2, i);
        }
    }
    std::vector<std::pair<double, int>> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    // куча выдаёт в обратном порядке, сортируем по возрастанию расстояния
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return result;
}
