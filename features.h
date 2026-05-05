#ifndef FEATURES_H
#define FEATURES_H

#include <array>
#include <string>

#include "wav_reader.h"

constexpr int FEATURE_DIM = 7;

// Индексы признаков в векторе.
enum FeatureIdx {
    F_DURATION = 0,
    F_RMS_MEAN = 1,
    F_RMS_VAR = 2,
    F_ZCR_VAR = 3,
    F_ZCR_MEAN = 4,
    F_BPM = 5,
    F_CENTROID = 6,
};

using FeatureVec = std::array<double, FEATURE_DIM>;

// Извлекает 7 признаков из WAV-данных.
FeatureVec extract_features(const WavData& wav);

// Извлекает признаки из файла. При ошибке возвращает false.
bool extract_features_from_file(const std::string& path, FeatureVec& out,
                                std::string& error);

const char* feature_name(int i);

#endif
