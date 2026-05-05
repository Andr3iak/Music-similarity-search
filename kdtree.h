#ifndef KDTREE_H
#define KDTREE_H

#include <memory>
#include <utility>
#include <vector>

#include "features.h"
#include "metrics.h"

struct KdNode {
    FeatureVec point{};
    int idx = -1;
    int axis = 0;
    std::unique_ptr<KdNode> left;
    std::unique_ptr<KdNode> right;
};

class KdTree {
   public:
    void build(const std::vector<FeatureVec>& points);
    std::vector<std::pair<double, int>> knn(const FeatureVec& query,
                                            const Weights& w, int k) const;

   private:
    std::unique_ptr<KdNode> root_;
};

#endif
