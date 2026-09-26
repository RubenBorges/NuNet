#pragma once
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <simdjson.h>
#include <debug.hpp>
#include <compare>
#include <concepts>
#include <type_traits>

namespace fs = std::filesystem;
using BitMasksType = std::ctype_base::mask;

namespace bpy {
#include <utility>

template <std::unsigned_integral T>
struct resolution {
    T columns{0};
    T rows{0};
    T channels{0};

    // Constructors
    resolution() = default;
    resolution(T c, T r, T ch) : columns(c), rows(r), channels(ch) {}

    // Copy and Move Constructors
    resolution(const resolution&) = default; 
    resolution(resolution&& other) noexcept : columns(other.columns), rows(other.rows), channels(other.channels) {}

        
    resolution& operator=(resolution rhs) noexcept {        // Unified Assignment Operator (The Copy-and-Swap Idiom)
        swap(*this, rhs); 
        return *this;  
    }

    auto operator<=>(const resolution& other) const = default; // Spaceship Operator

    /*/=========FRIEND-FUNCTIONS=========/*/
    friend std::strong_ordering current(const resolution& first, const resolution& second) {return first <=> second;} // Friendly Spaceship Wrapper

    friend void swap(resolution& first, resolution& second) noexcept { using std::swap;
        swap(first.columns, second.columns), swap(first.rows, second.rows), swap(first.channels, second.channels);
    }
};

class DataLoader {
using uint = std::uint8_t;
    resolution<uint> nativeResolution{0, 0, 0}; // Add safe defaults
    
public:
    dbglog* log{nullptr};
    std::size_t available_image_count = 0;

    DataLoader() = delete;
    DataLoader(dbglog& log_);

    // --- Image Loading API ---
    // Note: Replaced raw size_t with your custom resolution template to keep signatures clean!
    bool load_images_from_directory(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, size_t target_count, resolution<std::uint8_t> res);
    bool load_images_from_directory(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, size_t target_count, size_t rows, size_t cols, size_t channels);
    bool load_images_with_map(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, const std::unordered_map<std::string, float>& label_map, size_t target_count, resolution<std::uint8_t> res);
    bool load_images_with_map(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, const std::unordered_map<std::string, float>& label_map, size_t target_count, size_t rows, size_t cols, size_t channels);
    bool load_preprocessed_binaries(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& images_bin_path, const std::string& labels_bin_path);

    std::unordered_map<std::string, float> parse_flir_labels_simdjson(const std::string& json_path);
    std::unordered_map<std::string, float> parse_flir_v2_thermal_labels(const std::string& json_path);
    
    void shuffle_dataset(std::vector<float>& data_buffer, std::vector<float>& label_buffer, resolution<std::uint8_t> res);
    void shuffle_dataset(std::vector<float>& data_buffer, std::vector<float>& label_buffer, size_t rows, size_t cols, size_t channels);

    // --- Serialization API ---
    
    bool save_onednn_model(const std::string& filepath, const std::vector<float>& conv_w, const std::vector<float>& conv_b, const std::vector<float>& fc_w, const std::vector<float>& fc_b);
    bool save_onednn_model(std::string_view filepath, const std::vector<float>& conv_w, const std::vector<float>& conv_b, const std::vector<float>& fc_w, const std::vector<float>& fc_b);
    bool load_onednn_model(std::string_view path, std::vector<float>& conv_w, std::vector<float>& conv_b, std::vector<float>& fc_w, std::vector<float>& fc_b, std::chrono::hh_mm_ss<std::chrono::nanoseconds> log_time);

    // --- Resolution Getters (returns clean references) ---
    resolution<std::uint8_t>& getNativeResolution() noexcept { return nativeResolution; }
    constexpr const resolution<std::uint8_t>& getNativeResolution() const noexcept { return nativeResolution; }

    // --- Setter Functions for initializing and altering the recorded nativeResolution ---
    template <std::unsigned_integral T> void     Rows(T     rowCount) {nativeResolution.rows     = rowCount    > 0 ? static_cast<uint>     (rowCount) :1;}
    template <std::unsigned_integral T> void     Cols(T     colCount) {nativeResolution.columns  = colCount    > 0 ? static_cast<uint>     (colCount) :1;}
    template <std::unsigned_integral T> void Channels(T channelCount) {nativeResolution.channels = channelCount> 0 ? static_cast<uint> (channelCount) :1;}
};
} // namespace bpy
