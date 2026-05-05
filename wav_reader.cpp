#include "wav_reader.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

// Читаем little-endian целые числа из потока.
bool read_u32_le(std::ifstream& f, uint32_t& v) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char*>(b), 4)) return false;
    v = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) |
        (uint32_t(b[3]) << 24);
    return true;
}

bool read_u16_le(std::ifstream& f, uint16_t& v) {
    unsigned char b[2];
    if (!f.read(reinterpret_cast<char*>(b), 2)) return false;
    v = uint16_t(b[0]) | (uint16_t(b[1]) << 8);
    return true;
}

bool read_tag(std::ifstream& f, char tag[4]) {
    return static_cast<bool>(f.read(tag, 4));
}

}  // namespace

bool read_wav(const std::string& path, WavData& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "не удалось открыть файл: " + path;
        return false;
    }

    char riff[4];
    uint32_t riff_size = 0;
    char wave[4];
    if (!read_tag(f, riff) || !read_u32_le(f, riff_size) ||
        !read_tag(f, wave)) {
        error = "повреждённый WAV-заголовок";
        return false;
    }
    if (std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(wave, "WAVE", 4) != 0) {
        error = "файл не является RIFF/WAVE";
        return false;
    }

    bool fmt_seen = false;
    uint16_t audio_format = 0, num_channels = 0, block_align = 0,
             bits_per_sample = 0;
    uint32_t sample_rate = 0, byte_rate = 0;

    // Итеративно перебираем чанки, ищем "fmt " и "data".
    while (f) {
        char tag[4];
        uint32_t chunk_size = 0;
        if (!read_tag(f, tag)) break;
        if (!read_u32_le(f, chunk_size)) break;

        if (std::memcmp(tag, "fmt ", 4) == 0) {
            std::streampos start = f.tellg();
            if (!read_u16_le(f, audio_format) ||
                !read_u16_le(f, num_channels) ||
                !read_u32_le(f, sample_rate) || !read_u32_le(f, byte_rate) ||
                !read_u16_le(f, block_align) ||
                !read_u16_le(f, bits_per_sample)) {
                error = "не удалось прочитать fmt-чанк";
                return false;
            }
            fmt_seen = true;
            // Пропустить хвост fmt-чанка (если есть extra-поля).
            std::streamoff consumed = f.tellg() - start;
            std::streamoff to_skip =
                static_cast<std::streamoff>(chunk_size) - consumed;
            if (to_skip > 0) f.seekg(to_skip, std::ios::cur);
            // Выравнивание до чётного байта.
            if (chunk_size & 1u) f.seekg(1, std::ios::cur);
        } else if (std::memcmp(tag, "data", 4) == 0) {
            if (!fmt_seen) {
                error = "data-чанк до fmt-чанка";
                return false;
            }
            if (audio_format != 1) {
                error = "поддерживается только PCM (AudioFormat=1)";
                return false;
            }
            if (bits_per_sample != 16) {
                error = "поддерживается только 16-bit PCM";
                return false;
            }
            if (num_channels != 1 && num_channels != 2) {
                error = "поддерживается только mono или stereo";
                return false;
            }

            const uint32_t bytes_per_sample = bits_per_sample / 8;
            const uint32_t frame_bytes = bytes_per_sample * num_channels;
            const uint32_t num_frames = chunk_size / frame_bytes;

            std::vector<int16_t> raw(
                static_cast<size_t>(num_frames) * num_channels);
            if (!f.read(reinterpret_cast<char*>(raw.data()),
                        static_cast<std::streamsize>(raw.size()) *
                            sizeof(int16_t))) {
                error = "ошибка чтения data-чанка";
                return false;
            }

            out.sample_rate = static_cast<int>(sample_rate);
            out.num_channels = num_channels;
            out.bits_per_sample = bits_per_sample;
            out.samples.resize(num_frames);

            constexpr double kScale = 1.0 / 32768.0;
            if (num_channels == 1) {
                for (uint32_t i = 0; i < num_frames; ++i) {
                    out.samples[i] = raw[i] * kScale;
                }
            } else {
                // stereo -> усредняем каналы
                for (uint32_t i = 0; i < num_frames; ++i) {
                    double l = raw[2 * i] * kScale;
                    double r = raw[2 * i + 1] * kScale;
                    out.samples[i] = 0.5 * (l + r);
                }
            }
            return true;
        } else {
            // неизвестный чанк — пропускаем
            std::streamoff skip = chunk_size + (chunk_size & 1u);
            f.seekg(skip, std::ios::cur);
        }
    }

    error = "data-чанк не найден";
    return false;
}
