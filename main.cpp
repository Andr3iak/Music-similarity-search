#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "database.h"
#include "features.h"
#include "kdtree.h"
#include "linear_search.h"
#include "metrics.h"
#include "wav_reader.h"

namespace {

const std::string kDbPath = "database.csv";

// нормализация min-max по всей базе + запросу вместе
// важно нормировать совместно, иначе запрос будет в другом масштабе
void minmax_normalize(std::vector<FeatureVec>& all) {
    if (all.empty()) return;
    FeatureVec mn = all[0];
    FeatureVec mx = all[0];
    for (const auto& v : all) {
        for (int i = 0; i < FEATURE_DIM; ++i) {
            if (v[i] < mn[i]) mn[i] = v[i];
            if (v[i] > mx[i]) mx[i] = v[i];
        }
    }
    for (auto& v : all) {
        for (int i = 0; i < FEATURE_DIM; ++i) {
            double range = mx[i] - mn[i];
            v[i] = (range > 0.0) ? (v[i] - mn[i]) / range : 0.0;
        }
    }
}

void print_menu() {
    std::cout << "\n=== Music Recommender ===\n"
              << "1. Добавить трек в базу\n"
              << "2. Найти похожие треки\n"
              << "3. Показать базу\n"
              << "4. Выход\n"
              << "Выбор: ";
}

std::string read_line() {
    std::string s;
    std::getline(std::cin, s);
    return s;
}

void cmd_add(Database& db) {
    std::cout << "Путь к WAV: ";
    std::string path = read_line();

    FeatureVec f{};
    std::string err;
    if (!extract_features_from_file(path, f, err)) {
        std::cout << "Ошибка: " << err << "\n";
        return;
    }

    std::cout << "Имя трека (для базы): ";
    std::string name = read_line();
    if (name.empty()) name = path;

    TrackRecord rec;
    rec.name = name;
    rec.features = f;

    if (!db.append_to_file(kDbPath, rec, err)) {
        std::cout << "Ошибка записи в БД: " << err << "\n";
        return;
    }
    db.append(rec);
    std::cout << "Добавлено. Признаки:\n";
    for (int i = 0; i < FEATURE_DIM; ++i) {
        std::cout << "  " << feature_name(i) << " = " << f[i] << "\n";
    }
}

void cmd_show(const Database& db) {
    if (db.empty()) {
        std::cout << "База пуста.\n";
        return;
    }
    std::cout << std::left << std::setw(30) << "name";
    for (int i = 0; i < FEATURE_DIM; ++i) {
        std::cout << std::setw(14) << feature_name(i);
    }
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& r : db.records()) {
        std::cout << std::left << std::setw(30) << r.name;
        for (int i = 0; i < FEATURE_DIM; ++i) {
            std::cout << std::setw(14) << r.features[i];
        }
        std::cout << "\n";
    }
    std::cout.unsetf(std::ios::fixed);
}

void cmd_search(const Database& db) {
    if (db.empty()) {
        std::cout << "База пуста — добавьте треки.\n";
        return;
    }
    std::cout << "Путь к WAV-запросу: ";
    std::string qpath = read_line();

    FeatureVec qf{};
    std::string err;
    if (!extract_features_from_file(qpath, qf, err)) {
        std::cout << "Ошибка: " << err << "\n";
        return;
    }

    std::cout << "Введите k: ";
    std::string ks = read_line();
    int k = 0;
    try {
        k = std::stoi(ks);
    } catch (...) {
        std::cout << "Некорректное k.\n";
        return;
    }
    if (k <= 0) {
        std::cout << "k должно быть > 0.\n";
        return;
    }
    if (k > static_cast<int>(db.size())) k = static_cast<int>(db.size());

    std::cout << "Веса w[7] через пробел (Enter — все 1.0): ";
    std::string wline = read_line();
    Weights w;
    w.fill(1.0);
    if (!wline.empty()) {
        std::istringstream iss(wline);
        for (int i = 0; i < FEATURE_DIM; ++i) {
            if (!(iss >> w[i])) {
                std::cout << "Некорректные веса, использую 1.0.\n";
                w.fill(1.0);
                break;
            }
        }
    }

    // Нормализация: база + запрос.
    std::vector<FeatureVec> all;
    all.reserve(db.size() + 1);
    for (const auto& r : db.records()) all.push_back(r.features);
    all.push_back(qf);
    minmax_normalize(all);

    FeatureVec qn = all.back();
    all.pop_back();

    using clk = std::chrono::high_resolution_clock;

    auto t1 = clk::now();
    auto lin_res = linear_knn(all, qn, w, k);
    auto t2 = clk::now();

    KdTree tree;
    tree.build(all);
    auto t3 = clk::now();
    auto kd_res = tree.knn(qn, w, k);
    auto t4 = clk::now();

    // конвертируем duration в миллисекунды
    auto ms = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    std::cout << "\n--- LINEAR SCAN ---\n";
    for (const auto& p : lin_res) {
        std::cout << "  " << db.records()[p.second].name
                  << "  dist=" << std::sqrt(p.first) << "\n";
    }
    std::cout << "Время: " << ms(t2 - t1) << " ms\n";

    std::cout << "\n--- KD-TREE ---\n";
    for (const auto& p : kd_res) {
        std::cout << "  " << db.records()[p.second].name
                  << "  dist=" << std::sqrt(p.first) << "\n";
    }
    std::cout << "Время поиска: " << ms(t4 - t3) << " ms";
    std::cout << " (построение: " << ms(t3 - t2) << " ms)\n";

    // проверяем что оба алгоритма нашли одно и то же
    // сортируем по индексу т.к. при одинаковых расстояниях порядок может отличаться
    auto extract_set = [](const std::vector<std::pair<double, int>>& v) {
        std::vector<int> idx;
        for (const auto& p : v) idx.push_back(p.second);
        std::sort(idx.begin(), idx.end());
        return idx;
    };
    auto lin_set = extract_set(lin_res);
    auto kd_set = extract_set(kd_res);
    if (lin_set == kd_set) {
        std::cout << "Результаты обоих алгоритмов совпадают.\n";
    } else {
        std::cout << "ВНИМАНИЕ: результаты отличаются!\n";
        bool dist_match = lin_res.size() == kd_res.size();
        for (size_t i = 0; dist_match && i < lin_res.size(); ++i) {
            if (std::abs(lin_res[i].first - kd_res[i].first) > 1e-9)
                dist_match = false;
        }
        // расстояния должны совпадать даже если порядок разный
        assert(dist_match && "Расстояния линейного поиска и k-d дерева должны совпадать");
    }
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    Database db;
    std::string err;
    if (!db.load(kDbPath, err)) {
        std::cout << "[info] " << err << " — будет создана новая база.\n";
    } else {
        std::cout << "Загружено треков: " << db.size() << "\n";
    }

    while (true) {
        print_menu();
        std::string choice = read_line();
        if (choice == "1") {
            cmd_add(db);
        } else if (choice == "2") {
            cmd_search(db);
        } else if (choice == "3") {
            cmd_show(db);
        } else if (choice == "4" || choice == "q" || choice == "exit") {
            break;
        } else if (choice.empty()) {
            continue;
        } else {
            std::cout << "Неизвестная команда.\n";
        }
    }
    return 0;
}
