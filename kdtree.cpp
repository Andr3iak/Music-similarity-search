#include "kdtree.h"

#include <algorithm>
#include <queue>

namespace {

using Heap = std::priority_queue<std::pair<double, int>>;

// строим k-d дерево рекурсивно
// на каждом уровне чередуем ось разбиения (depth % FEATURE_DIM)
// [lo, hi) — текущий диапазон индексов
std::unique_ptr<KdNode> build_recursive(const std::vector<FeatureVec>& pts,
                                        std::vector<int>& indices, int lo,
                                        int hi, int depth) {
    if (lo >= hi) return nullptr;
    int axis = depth % FEATURE_DIM;
    int mid = lo + (hi - lo) / 2;

    // nth_element ставит медиану в позицию mid без полной сортировки — O(n)
    std::nth_element(indices.begin() + lo, indices.begin() + mid,
                     indices.begin() + hi,
                     [&](int a, int b) {
                         return pts[a][axis] < pts[b][axis];
                     });

    auto node = std::make_unique<KdNode>();
    int pi = indices[mid];
    node->point = pts[pi];
    node->idx = pi;
    node->axis = axis;
    node->left = build_recursive(pts, indices, lo, mid, depth + 1);
    node->right = build_recursive(pts, indices, mid + 1, hi, depth + 1);
    return node;
}

void knn_recursive(const KdNode* node, const FeatureVec& q, const Weights& w,
                   int k, Heap& heap) {
    if (!node) return;

    double d2 = weighted_sq_distance(node->point, q, w);
    // heap — max-куча, на вершине самый дальний из k кандидатов
    if (static_cast<int>(heap.size()) < k) {
        heap.emplace(d2, node->idx);
    } else if (d2 < heap.top().first) {
        heap.pop();
        heap.emplace(d2, node->idx);
    }

    int axis = node->axis;
    double diff = q[axis] - node->point[axis];
    const KdNode* near_child = (diff < 0.0) ? node->left.get() : node->right.get();
    const KdNode* far_child = (diff < 0.0) ? node->right.get() : node->left.get();

    knn_recursive(near_child, q, w, k, heap);

    // проверяем дальнее поддерево только если оно может содержать более близкие точки
    // (расстояние до разделяющей гиперплоскости меньше текущего k-го)
    double axis_d2 = w[axis] * diff * diff;
    if (static_cast<int>(heap.size()) < k || axis_d2 < heap.top().first) {
        knn_recursive(far_child, q, w, k, heap);
    }
}

}  // namespace

void KdTree::build(const std::vector<FeatureVec>& points) {
    root_.reset();
    if (points.empty()) return;
    std::vector<int> indices(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) indices[i] = i;
    root_ = build_recursive(points, indices, 0,
                            static_cast<int>(points.size()), 0);
}

std::vector<std::pair<double, int>> KdTree::knn(const FeatureVec& query,
                                                const Weights& w,
                                                int k) const {
    Heap heap;
    knn_recursive(root_.get(), query, w, k, heap);
    std::vector<std::pair<double, int>> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return result;
}
