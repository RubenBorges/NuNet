#include <debug.hpp>


constexpr std::string_view dbglog::to_string(lvl level) const {
    switch (level) {
        case info:  return "INFO";
        case alert: return "ALERT";
        case WARN:  return "WARNING";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

dbglog::dbglog(std::chrono::year_month_day date, const std::filesystem::path& log_path) : log_date(date) {
    file.open(log_path, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::clog << "Failed to open debug.log!\n";
    }
}

void dbglog::operator()(std::chrono::hh_mm_ss<std::chrono::nanoseconds> time, lvl level, std::string_view message) {
    if (!file.is_open()) return;
    
    std::println(file, "{0} {1}: {2} -- {3}", log_date, time, message, to_string(level));
}


