#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include <vector>

#include "features.h"

struct TrackRecord {
    std::string name;
    FeatureVec features{};
};

class Database {
   public:
    bool load(const std::string& path, std::string& error);
    bool save(const std::string& path, std::string& error) const;
    void append(const TrackRecord& rec);

    // Дописывает строку в существующий CSV (без перезаписи).
    bool append_to_file(const std::string& path, const TrackRecord& rec,
                        std::string& error) const;

    const std::vector<TrackRecord>& records() const { return records_; }
    std::vector<TrackRecord>& records() { return records_; }
    bool empty() const { return records_.empty(); }
    size_t size() const { return records_.size(); }

   private:
    std::vector<TrackRecord> records_;
};

#endif
