#pragma once 
#include <dnnl.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string_view>
#include <print>
#include <vector>
#include <cmath>
#include <numeric>
#include <filesystem>


class dbglog {
public:
    enum lvl : std::uint8_t { info = 0, alert = 1, WARN = 2, ERROR = 3 };

    std::fstream file;
    std::chrono::year_month_day log_date;

    constexpr std::string_view to_string(lvl level) const ;

    dbglog(std::chrono::year_month_day date, const std::filesystem::path& log_path);

void operator()(std::chrono::hh_mm_ss<std::chrono::nanoseconds> time, lvl level, std::string_view message) ;
};

