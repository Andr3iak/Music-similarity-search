#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "database.h"
#include "features.h"
#include "kdtree.h"
#include "linear_search.h"
#include "metrics.h"

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// =========================================================
// Нормализация min-max
// Параметры считаются по обучающей выборке, потом применяются
// к запросу — так нельзя «подглядывать» в данные запроса
// =========================================================

struct NormParams {
    FeatureVec mn{};
    FeatureVec mx{};
};

NormParams compute_norm(const std::vector<FeatureVec>& vecs) {
    NormParams p;
    // нужно явно инициализировать, иначе min/max будут мусором
    p.mn.fill(std::numeric_limits<double>::max());
    p.mx.fill(std::numeric_limits<double>::lowest());
    for (const auto& v : vecs)
        for (int i = 0; i < FEATURE_DIM; ++i) {
            if (v[i] < p.mn[i]) p.mn[i] = v[i];
            if (v[i] > p.mx[i]) p.mx[i] = v[i];
        }
    return p;
}

// применяет уже посчитанные min/max к одному вектору (для нормализации запроса)
FeatureVec apply_norm(const FeatureVec& v, const NormParams& p) {
    FeatureVec out;
    for (int i = 0; i < FEATURE_DIM; i++) {
        double range = p.mx[i] - p.mn[i];
        out[i] = range > 0.0 ? (v[i] - p.mn[i]) / range : 0.0;
    }
    return out;
}

std::vector<FeatureVec> normalize_all(const std::vector<FeatureVec>& vecs, const NormParams& p) {
    std::vector<FeatureVec> result;
    result.reserve(vecs.size());
    for (const auto& v : vecs)
        result.push_back(apply_norm(v, p));
    return result;
}

// =========================================================
// Вспомогательное
// =========================================================

static const std::vector<std::string> kGenres = {
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
};

static const char* kBenchDb  = "bench_database.csv";
static const char* kDataRoot = "DATA/genres_original";

// жанр — это то что до первого слеша в имени ("blues/blues.00000.wav" -> "blues")
std::string genre_of(const std::string& name) {
    size_t pos = name.find('/');
    if (pos == std::string::npos) return name;
    return name.substr(0, pos);
}

double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct WeightSet {
    std::string name;
    Weights w;
};

// =========================================================
// Шаг 0 — загрузка / сборка базы
// =========================================================

std::vector<TrackRecord> step0_database() {
    std::cout << "=== STEP 0: DATABASE ===\n";

    // если кэш есть — не тратим время на пересчёт фич
    {
        std::ifstream probe(kBenchDb);
        if (probe) {
            Database db;
            std::string err;
            if (db.load(kBenchDb, err)) {
                auto& recs = db.records();
                std::cout << "Загружено из кэша " << kBenchDb << ": " << recs.size() << " треков\n";
                std::map<std::string, int> cnt;
                for (const auto& r : recs)
                    cnt[genre_of(r.name)]++;
                for (const auto& g : kGenres)
                    std::cout << "  " << std::left << std::setw(12) << g << ": " << cnt[g] << "\n";
                return recs;
            }
            // кэш побит или пустой — пересобираем
            std::cout << "Кэш повреждён (" << err << "), пересобираем...\n";
        }
    }

    // TODO: можно было бы параллелить по жанрам, но пока так
    std::vector<TrackRecord> result;

    for (const auto& genre : kGenres) {
        fs::path dir = fs::path(kDataRoot) / genre;
        if (!fs::is_directory(dir)) {
            std::cout << "  " << genre << ": папка не найдена, пропускаем\n";
            continue;
        }

        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".wav")
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());

        int ok = 0;
        for (const auto& fp : files) {
            FeatureVec fv;
            std::string err;
            if (!extract_features_from_file(fp.string(), fv, err)) {
                std::cerr << "  Пропуск " << fp.filename().string() << ": " << err << "\n";
                continue;
            }
            TrackRecord rec;
            rec.name = genre + "/" + fp.filename().string();
            rec.features = fv;
            result.push_back(rec);
            ok++;
        }
        std::cout << "  " << std::left << std::setw(12) << genre << ": " << ok << " треков\n";
    }

    std::cout << "Итого: " << result.size() << " треков\n";

    // сохраняем кэш
    Database db;
    for (const auto& r : result) db.append(r);
    std::string err;
    if (!db.save(kBenchDb, err))
        std::cerr << "Предупреждение: не удалось сохранить кэш: " << err << "\n";
    else
        std::cout << "Кэш сохранён в " << kBenchDb << "\n";

    return result;
}

// =========================================================
// Шаг 1 — масштабируемость по N
// =========================================================

// возвращает {linear_us, kdtree_us} при N=1000 для итоговой сводки
std::pair<double, double> step1_scaling(const std::vector<TrackRecord>& all) {
    std::cout << "\n=== EXPERIMENT 1: SCALING ===\n";

    int total = (int)all.size();
    std::vector<int> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(42);  // фиксированный seed для воспроизводимости
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<int> Ns = {50, 100, 200, 500, 1000};
    const int kQ = 20;
    const int kK = 10;

    Weights weq;
    weq.fill(1.0);  // равные веса для эксперимента со скоростью

    std::ofstream csv("bench_scaling.csv");
    csv << "N,linear_us,kdtree_build_ms,kdtree_query_us\n";

    // шапка таблицы
    std::cout << std::right
              << std::setw(6)  << "N"
              << std::setw(14) << "linear_us"
              << std::setw(18) << "kdtree_build_ms"
              << std::setw(16) << "kdtree_us\n";
    std::cout << std::string(54, '-') << "\n";

    double ret_lin = 0.0, ret_kd = 0.0;

    for (int N : Ns) {
        // подвыборка — первые N из перемешанного порядка
        std::vector<FeatureVec> sub;
        sub.reserve(N);
        for (int i = 0; i < N; i++)
            sub.push_back(all[order[i]].features);

        // запросы — следующие kQ записей (не входят в подвыборку, если N < 980)
        int q0 = std::min(N, total - kQ);
        std::vector<FeatureVec> qraw;
        for (int i = 0; i < kQ; i++)
            qraw.push_back(all[order[q0 + i]].features);

        // нормализуем подвыборку и запросы совместно
        std::vector<FeatureVec> combined(sub);
        combined.insert(combined.end(), qraw.begin(), qraw.end());
        NormParams p = compute_norm(combined);

        auto sub_n = normalize_all(sub, p);
        std::vector<FeatureVec> q_n;
        for (const auto& v : qraw)
            q_n.push_back(apply_norm(v, p));

        // --- линейный поиск ---
        double lin_total = 0.0;
        for (const auto& q : q_n) {
            auto t = Clock::now();
            linear_knn(sub_n, q, weq, kK);
            lin_total += us_since(t);
        }
        double lin_avg = lin_total / kQ;

        // --- k-d дерево: сначала строим, потом запрашиваем ---
        auto t_build = Clock::now();
        KdTree tree;
        tree.build(sub_n);
        double build_ms = ms_since(t_build);

        double kd_total = 0.0;
        for (const auto& q : q_n) {
            auto t = Clock::now();
            tree.knn(q, weq, kK);
            kd_total += us_since(t);
        }
        double kd_avg = kd_total / kQ;

        if (N == 1000) {
            ret_lin = lin_avg;
            ret_kd  = kd_avg;
        }

        csv << N << "," << std::fixed << std::setprecision(2)
            << lin_avg << "," << build_ms << "," << kd_avg << "\n";

        std::cout << std::right << std::setw(6) << N
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << lin_avg
                  << std::setw(18) << build_ms
                  << std::setw(16) << kd_avg << "\n";
    }

    std::cout << "Результаты сохранены в bench_scaling.csv\n";
    return {ret_lin, ret_kd};
}

// =========================================================
// Шаг 2 — качество рекомендаций (genre@10)
// =========================================================

int step2_quality(const std::vector<TrackRecord>& all, const std::vector<WeightSet>& wsets) {
    std::cout << "\n=== EXPERIMENT 2: RECOMMENDATION QUALITY (genre@10) ===\n";

    int N = (int)all.size();
    const int kK = 10;

    // собираем векторы признаков и метки жанров
    std::vector<FeatureVec> fvs;
    fvs.reserve(N);
    for (const auto& r : all) fvs.push_back(r.features);

    std::vector<std::string> gens;
    gens.reserve(N);
    for (const auto& r : all) gens.push_back(genre_of(r.name));

    // нормализуем всю базу целиком — параметры фиксируются по базе
    NormParams p = compute_norm(fvs);
    auto nfvs = normalize_all(fvs, p);

    std::ofstream csv("bench_quality.csv");
    csv << "weights_name,genre,genre_at_10\n";

    int best_wi = 0;
    double best_avg = -1.0;
    std::vector<double> avgs;
    // gscores[wi][genre] = средний genre@10 для данного набора весов
    std::vector<std::map<std::string, double>> gscores(wsets.size());

    for (int wi = 0; wi < (int)wsets.size(); wi++) {
        std::cout << "  Считаем " << wsets[wi].name << "...";
        std::cout.flush();  // чтобы точки появлялись по мере счёта

        std::map<std::string, double> gsum;
        std::map<std::string, int> gcnt;
        for (const auto& g : kGenres) { gsum[g] = 0.0; gcnt[g] = 0; }
        double total_score = 0.0;

        // для каждого трека делаем запрос к остальным 999
        // запрашиваем kK+1 чтобы потом выкинуть сам трек из результата
        for (int i = 0; i < N; i++) {
            auto res = linear_knn(nfvs, nfvs[i], wsets[wi].w, kK + 1);

            int same_genre = 0, counted = 0;
            for (auto& [d, idx] : res) {
                if (idx == i) continue;  // пропускаем сам трек (расстояние = 0)
                if (gens[idx] == gens[i]) same_genre++;
                if (++counted >= kK) break;
            }

            double score = (double)same_genre / kK;
            gsum[gens[i]] += score;
            gcnt[gens[i]]++;
            total_score += score;
        }

        double avg = total_score / N;
        avgs.push_back(avg);
        if (avg > best_avg) { best_avg = avg; best_wi = wi; }

        for (const auto& g : kGenres) {
            double s = gcnt[g] > 0 ? gsum[g] / gcnt[g] : 0.0;
            gscores[wi][g] = s;
            csv << wsets[wi].name << "," << g << ","
                << std::fixed << std::setprecision(4) << s << "\n";
        }
        // пишем среднее по всем жанрам в csv тоже
        csv << wsets[wi].name << ",AVERAGE,"
            << std::fixed << std::setprecision(4) << avg << "\n";

        std::cout << " avg=" << std::fixed << std::setprecision(4) << avg << "\n";
    }

    // сводная таблица в консоль
    const int CW = 10;
    int table_width = 14 + CW * (int)wsets.size();

    std::cout << "\n" << std::left << std::setw(14) << "Жанр";
    for (const auto& ws : wsets)
        std::cout << std::right << std::setw(CW) << ws.name;
    std::cout << "\n" << std::string(table_width, '-') << "\n";

    for (const auto& g : kGenres) {
        std::cout << std::left << std::setw(14) << g;
        for (int wi = 0; wi < (int)wsets.size(); wi++)
            std::cout << std::right << std::fixed << std::setprecision(4)
                      << std::setw(CW) << gscores[wi][g];
        std::cout << "\n";
    }
    std::cout << std::string(table_width, '-') << "\n";
    std::cout << std::left << std::setw(14) << "СРЕДНЕЕ";
    for (double a : avgs)
        std::cout << std::right << std::fixed << std::setprecision(4) << std::setw(CW) << a;
    std::cout << "\n";

    std::cout << "\nЛучший набор весов: " << wsets[best_wi].name
              << "  (avg genre@10 = " << std::fixed << std::setprecision(4) << best_avg << ")\n";
    std::cout << "Результаты сохранены в bench_quality.csv\n";

    return best_wi;
}

// =========================================================
// Шаг 3 — примеры выдачи для 3 эталонных треков
// =========================================================

void step3_examples(const std::vector<TrackRecord>& all, const WeightSet& best) {
    std::cout << "\n=== EXPERIMENT 3: RECOMMENDATION EXAMPLES ===\n";
    std::cout << "Веса: " << best.name << "\n\n";

    // нормализуем так же как в шаге 2
    std::vector<FeatureVec> fvs;
    for (const auto& r : all) fvs.push_back(r.features);
    NormParams p = compute_norm(fvs);
    auto nfvs = normalize_all(fvs, p);

    // строим словарь имя -> индекс чтобы быстро находить эталоны
    std::map<std::string, int> name_idx;
    for (int i = 0; i < (int)all.size(); i++)
        name_idx[all[i].name] = i;

    const std::vector<std::string> refs = {
        "blues/blues.00000.wav",
        "classical/classical.00000.wav",
        "metal/metal.00000.wav",
    };

    std::ofstream txt("bench_examples.txt");
    txt << "Примеры рекомендаций (веса: " << best.name << ")\n";
    txt << std::string(60, '=') << "\n\n";

    // вывод в консоль и в файл одновременно
    auto out = [&](const std::string& line) {
        std::cout << line << "\n";
        txt << line << "\n";
    };

    for (const auto& ref : refs) {
        auto it = name_idx.find(ref);
        if (it == name_idx.end()) {
            out("НЕТ В БАЗЕ: " + ref);
            continue;
        }
        int qi = it->second;

        // запрашиваем 6 вместо 5 — сам трек тоже попадает в результат, его выкидываем
        auto res = linear_knn(nfvs, nfvs[qi], best.w, 6);

        out("Запрос: " + ref);
        int rank = 0;
        for (auto& [dsq, idx] : res) {
            if (idx == qi) continue;
            rank++;
            std::ostringstream ss;
            ss << "  " << rank << ". " << all[idx].name
               << "  dist=" << std::fixed << std::setprecision(6) << std::sqrt(dsq);
            out(ss.str());
            if (rank >= 5) break;
        }
        out("");
    }

    std::cout << "Сохранено в bench_examples.txt\n";
}

// =========================================================
// main
// =========================================================

int main() {
    SetConsoleOutputCP(CP_UTF8);

    auto all = step0_database();
    if (all.empty()) {
        std::cerr << "Треки не загружены, выход.\n";
        return 1;
    }

    auto [lin1000, kd1000] = step1_scaling(all);

    // наборы весов подбирались экспериментально:
    // w1 — базовый, w2-w4 — усиливаем признаки ритма и тембра
    const std::vector<WeightSet> wsets = {
        {"w1", {1, 1, 1, 1, 1, 1, 1}},  // равные веса
        {"w2", {1, 1, 1, 1, 1, 3, 2}},  // акцент на BPM и центроид
        {"w3", {1, 2, 1, 1, 2, 3, 2}},  // + RMS и ZCR
        {"w4", {1, 3, 2, 2, 3, 4, 3}},  // сильный акцент на динамику и ритм
    };

    int best_wi = step2_quality(all, wsets);
    step3_examples(all, wsets[best_wi]);

    // итоговая сводка
    std::cout << "\n=== ИТОГ ===\n";
    std::cout << "Лучший набор весов: " << wsets[best_wi].name << "\n";
    if (kd1000 > 0.0) {
        std::cout << "k-d дерево быстрее линейного при N=1000: "
                  << std::fixed << std::setprecision(1)
                  << lin1000 / kd1000 << "x\n";
    }

    return 0;
}
