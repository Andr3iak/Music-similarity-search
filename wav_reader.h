#ifndef WAV_READER_H
#define WAV_READER_H

#include <string>
#include <vector>

struct WavData {
    int sample_rate = 0;
    int num_channels = 0;
    int bits_per_sample = 0;
    // моно сигнал в диапазоне [-1.0, 1.0] (стерео усредняется)
    std::vector<double> samples;
};

// Читает PCM 16-bit WAV (mono/stereo). Возвращает false при ошибке.
bool read_wav(const std::string& path, WavData& out, std::string& error);

#endif
