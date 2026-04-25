// структура данных для представления треков

#ifndef RECOMMENDED_TRACK_HPP  // защита от лишней попытки компиляции
#define RECOMMENDED_TRACK_HPP

#include <array>
#include <cstddef>  // для удобных типов (size_t и др.)
#include <string>

namespace rec {

// вектор характеристик
constexpr std::size_t FEATURE_DIM = 8;

enum FeatureIndex : std::size_t {
    F_TEMPO = 0,
    F_ENERGY = 1,
    F_DRIVE = 2,
    FVALENCE = 3,
    F_LOUDNESS = 4,
    F_KEY = 5,
    F_MODE = 6,
    F_DURATION = 7
};

// Имена признаков (для вывода и заголовков CSV)
inline const char* feature_name(std::size_t i) {
    static const char* names[FEATURE_DIM] = {"tempo",   "energy",   "drive",
                                             "valence", "loudness", "key",
                                             "mode",    "duration"};
    return (i < FEATURE_DIM) ? names[i] : "?";
}

using FeatureVector = std::array<double, FEATURE_DIM>;

struct Track {
    int id = 0;
    std::string title;
    std::string artist;
    std::string genre;
    FeatureVector features{};
};

}  // namespace rec

#endif