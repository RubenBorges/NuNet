#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <simdjson.h>
#include <debug.hpp>

namespace fs = std::filesystem;

namespace bpy {

class DataLoader {
public:
    dbglog* log;
    size_t available_image_count = 0;

    DataLoader() = delete;
    DataLoader(dbglog& log_);

    bool load_images_from_directory(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, size_t target_count, size_t rows, size_t cols, size_t channels);
    
    bool load_images_with_map(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, const std::unordered_map<std::string, float>& label_map, size_t target_count, size_t rows, size_t cols, size_t channels);
    
    std::unordered_map<std::string, float> parse_flir_labels_simdjson(const std::string& json_path);

    std::unordered_map<std::string, float> parse_flir_v2_thermal_labels(const std::string& json_path);
    
    // Unified shuffling method
    void shuffle_dataset(std::vector<float>& data_buffer, std::vector<float>& label_buffer, size_t rows, size_t cols, size_t channels);

    bool save_onednn_model(const std::string& filepath, const std::vector<float>& conv_w, const std::vector<float>& conv_b, const std::vector<float>& fc_w, const std::vector<float>& fc_b);

    bool load_onednn_model(std::string_view path, std::vector<float>& conv_w, std::vector<float>& conv_b, std::vector<float>& fc_w, std::vector<float>& fc_b, std::chrono::hh_mm_ss<std::chrono::nanoseconds> log_time);

};
} // namespace bpy
