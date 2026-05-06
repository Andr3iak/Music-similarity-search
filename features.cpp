#include "features.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// полосовой фильтр 2-го порядка (биквад), Direct Form I
// f0 - центральная частота, Q - добротность
// возвращает суммарную энергию сигнала после фильтрации
double bandpass_energy(const std::vector<double>& x, double f0, double Q,
                       double Fs) {
    const double w0 = 2.0 * kPi * f0 / Fs;
    const double cw0 = std::cos(w0);
    const double sw0 = std::sin(w0);
    const double alpha = sw0 / (2.0 * Q);

    double b0 = alpha;
    double b1 = 0.0;
    double b2 = -alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cw0;
    double a2 = 1.0 - alpha;

    // нормируем коэффициенты на a0
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;

    double x1 = 0.0, x2 = 0.0;
    double y1 = 0.0, y2 = 0.0;
    double energy = 0.0;
    for (double xn : x) {
        double yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = xn;
        y2 = y1;
        y1 = yn;
        energy += yn * yn;
    }
    return energy;
}

// оценка темпа (BPM) через автокорреляцию огибающей энергии
// делим на фреймы по 512 сэмплов, считаем RMS каждого фрейма
// потом ищем период повторения в диапазоне 60-180 BPM
double estimate_bpm(const std::vector<double>& samples, int sample_rate) {
    constexpr int kFrame = 512;
    const int n_frames = static_cast<int>(samples.size()) / kFrame;
    if (n_frames < 4) return 0.0;

    std::vector<double> env(n_frames);
    for (int i = 0; i < n_frames; ++i) {
        double sum_sq = 0.0;
        const double* p = &samples[static_cast<size_t>(i) * kFrame];
        for (int j = 0; j < kFrame; ++j) sum_sq += p[j] * p[j];
        env[i] = std::sqrt(sum_sq / kFrame);
    }

    // вычитаем среднее — без этого автокорреляция смещена вверх
    double mean = 0.0;
    for (double v : env) mean += v;
    mean /= env.size();
    for (double& v : env) v -= mean;

    // переводим диапазон BPM в лаги (количество фреймов)
    const double frame_rate = static_cast<double>(sample_rate) / kFrame;
    int L_min = static_cast<int>(std::round(frame_rate * 60.0 / 180.0));
    int L_max = static_cast<int>(std::round(frame_rate * 60.0 / 60.0));
    if (L_min < 1) L_min = 1;
    if (L_max >= n_frames) L_max = n_frames - 1;
    if (L_min > L_max) return 0.0;

    // TODO: можно добавить параболическую интерполяцию для точности
    int best_lag = L_min;
    double best_corr = -1e300;
    for (int L = L_min; L <= L_max; ++L) {
        double c = 0.0;
        for (int i = 0; i + L < n_frames; ++i) c += env[i] * env[i + L];
        if (c > best_corr) {
            best_corr = c;
            best_lag = L;
        }
    }
    if (best_lag <= 0) return 0.0;
    return frame_rate / best_lag * 60.0;
}

}  // namespace

FeatureVec extract_features(const WavData& wav) {
    FeatureVec f{};
    if (wav.samples.empty() || wav.sample_rate <= 0) return f;

    const auto& s = wav.samples;
    const double Fs = static_cast<double>(wav.sample_rate);
    double peak_amp = 0.0;
    for (double v : s) {
        double av = std::abs(v);
        if (av > peak_amp) peak_amp = av;
    }

    // считаем RMS и ZCR по фреймам (размер фрейма 512 — стандарт для аудио)
    constexpr int kFrame = 512;
    const int n_frames = static_cast<int>(s.size()) / kFrame;

    double rms_mean = 0.0, rms_var = 0.0;
    double zcr_mean = 0.0, zcr_var = 0.0;

    if (n_frames > 0) {
        std::vector<double> rms_frames(n_frames);
        std::vector<double> zcr_frames(n_frames);
        const double frame_dur = static_cast<double>(kFrame) / Fs;

        for (int i = 0; i < n_frames; ++i) {
            const double* p = &s[static_cast<size_t>(i) * kFrame];
            double sum_sq = 0.0;
            long zc = 0;
            for (int j = 0; j < kFrame; ++j) {
                sum_sq += p[j] * p[j];
                if (j > 0 && ((p[j - 1] >= 0.0) != (p[j] >= 0.0))) ++zc;
            }
            rms_frames[i] = std::sqrt(sum_sq / kFrame);
            zcr_frames[i] = static_cast<double>(zc) / frame_dur;
        }

        // среднее
        for (int i = 0; i < n_frames; ++i) {
            rms_mean += rms_frames[i];
            zcr_mean += zcr_frames[i];
        }
        rms_mean /= n_frames;
        zcr_mean /= n_frames;

        // дисперсия (population variance, делим на N а не N-1)
        for (int i = 0; i < n_frames; ++i) {
            double dr = rms_frames[i] - rms_mean;
            double dz = zcr_frames[i] - zcr_mean;
            rms_var += dr * dr;
            zcr_var += dz * dz;
        }
        rms_var /= n_frames;
        zcr_var /= n_frames;
    }

    double bpm = estimate_bpm(s, wav.sample_rate);
    // std::cout << "[debug] bpm = " << bpm << "\n";

    // спектральный центроид через банк фильтров по октавным частотам
    // чем выше centroid - тем "ярче" звучит трек
    static const double freqs[8] = {125.0,  250.0,  500.0,  1000.0,
                                    2000.0, 4000.0, 8000.0, 16000.0};
    double num = 0.0, den = 0.0;
    for (double fk : freqs) {
        if (fk >= Fs * 0.5) continue;  // выше найквиста не идём
        double e = bandpass_energy(s, fk, 1.0, Fs);
        num += fk * e;
        den += e;
    }
    double centroid = (den > 0.0) ? num / den : 0.0;

    f[F_PEAK_AMP] = peak_amp;
    f[F_RMS_MEAN] = rms_mean;
    f[F_RMS_VAR] = rms_var;
    f[F_ZCR_VAR] = zcr_var;
    f[F_ZCR_MEAN] = zcr_mean;
    f[F_BPM] = bpm;
    f[F_CENTROID] = centroid;
    return f;
}

bool extract_features_from_file(const std::string& path, FeatureVec& out,
                                std::string& error) {
    WavData wav;
    if (!read_wav(path, wav, error)) return false;
    out = extract_features(wav);
    return true;
}

const char* feature_name(int i) {
    static const char* names[FEATURE_DIM] = {
        "peak_amplitude", "rms_mean", "rms_var",         "zcr_var",
        "zcr_mean",     "bpm",      "spectral_centroid"};
    return (i >= 0 && i < FEATURE_DIM) ? names[i] : "?";
}
