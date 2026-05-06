#include "database.h"

#include <fstream>
#include <sstream>

namespace {

const char* kHeader =
    "name,peak_amplitude,rms_mean,rms_var,zcr_var,zcr_mean,bpm,spectral_"
    "centroid";

// экранируем имя трека если в нём есть запятые или кавычки
std::string escape_csv(const std::string& s) {
    bool needs_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');  // "" = экранированная кавычка в CSV
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// разбиваем строку CSV на поля, учитываем кавычки
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else if (c == '"') {
                in_quotes = true;
            } else {
                cur.push_back(c);
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

std::string format_record(const TrackRecord& r) {
    std::ostringstream oss;
    oss.precision(10);  // 10 знаков — достаточно для double
    oss << escape_csv(r.name);
    for (int i = 0; i < FEATURE_DIM; ++i) {
        oss << "," << r.features[i];
    }
    return oss.str();
}

}  // namespace

bool Database::load(const std::string& path, std::string& error) {
    records_.clear();
    std::ifstream f(path);
    if (!f) {
        error = "не удалось открыть " + path;
        return false;
    }
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();  // Windows \r\n
        if (line.empty()) continue;
        if (first) {
            first = false;
            continue;  // пропускаем строку-заголовок
        }
        auto fields = split_csv(line);
        if (fields.size() != 1 + FEATURE_DIM) {
            error = "некорректное число полей в строке: " + line;
            return false;
        }
        TrackRecord rec;
        rec.name = fields[0];
        try {
            for (int i = 0; i < FEATURE_DIM; ++i)
                rec.features[i] = std::stod(fields[1 + i]);
        } catch (...) {
            error = "ошибка парсинга чисел в строке: " + line;
            return false;
        }
        records_.push_back(std::move(rec));
    }
    return true;
}

bool Database::save(const std::string& path, std::string& error) const {
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        error = "не удалось открыть для записи: " + path;
        return false;
    }
    f << kHeader << "\n";
    for (const auto& r : records_) f << format_record(r) << "\n";
    return true;
}

void Database::append(const TrackRecord& rec) { records_.push_back(rec); }

bool Database::append_to_file(const std::string& path, const TrackRecord& rec,
                              std::string& error) const {
    // если файл не существует — нужно добавить заголовок
    bool need_header = false;
    {
        std::ifstream test(path);
        if (!test) need_header = true;
    }
    std::ofstream f(path, std::ios::app);
    if (!f) {
        error = "не удалось открыть для дозаписи: " + path;
        return false;
    }
    if (need_header) f << kHeader << "\n";
    f << format_record(rec) << "\n";
    return true;
}
