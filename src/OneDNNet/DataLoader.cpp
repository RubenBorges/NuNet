#include <DataLoader.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <print>
#include <simdjson.h>
#include <random>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

using namespace bpy;
using namespace simdjson;

    
    DataLoader::DataLoader(dbglog& log_):log(&log_) {}

bool DataLoader::save_onednn_model(const std::string& filepath,const std::vector<float>& conv_w, const std::vector<float>& conv_b, const std::vector<float>& fc_w, const std::vector<float>& fc_b) {
    auto now = std::chrono::system_clock::now();
    auto current_day = std::chrono::floor<std::chrono::days>(now);
    std::chrono::hh_mm_ss<std::chrono::nanoseconds> log_time{now - current_day};

    std::ofstream out_file(filepath, std::ios::binary | std::ios::trunc);
    if (!out_file.is_open()) {
        std::println(std::cerr, "Error: Failed to open model export file path: {}", filepath);
        (*log)(log_time, dbglog::lvl::ERROR, "Failed to save model.");

        return false;
    }

    auto write_vector = [&](const std::vector<float>& vec) {
        size_t size = vec.size();
        out_file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out_file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    };

    // Sequentially stream flat representations out to disk
    write_vector(conv_w);
    write_vector(conv_b);
    write_vector(fc_w);
    write_vector(fc_b);

    out_file.close();
    std::println("Model weights successfully serialized to binary payload: {}", filepath);
    (*log)(log_time, dbglog::lvl::WARN, "Model saved successfully.");
    return true;
}

bool DataLoader::load_onednn_model(std::string_view path, std::vector<float>& conv_w, std::vector<float>& conv_b, std::vector<float>& fc_w, std::vector<float>& fc_b, std::chrono::hh_mm_ss<std::chrono::nanoseconds> log_time) {
    // 1. Proactively check if the file even exists
    if (!std::filesystem::exists(path)) {
        (*log)(log_time, dbglog::lvl::ERROR, "Failed to LOAD model: File does not exist at target path.");
        return false;
    }

    // 2. Open binary file for parsing input
    std::ifstream file{std::filesystem::path(path), std::ios::in | std::ios::binary};

    // 3. Fallback check if path exists but handle fails to bind
    if (!file.is_open()) {
        (*log)(log_time, dbglog::lvl::ERROR, "Failed to LOAD model: Could not acquire file stream handle.");
        return false;
    }

    try {
        auto read_vector = [&file](std::vector<float>& values) {
            size_t size = 0;
            if (!file.read(reinterpret_cast<char*>(&size), sizeof(size)) ||
                size > values.max_size() ||
                size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
                return false;
            }

            values.resize(size);
            return size == 0 || static_cast<bool>(file.read(
                reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(size * sizeof(float))));
        };

        if (!read_vector(conv_w) || !read_vector(conv_b) ||
            !read_vector(fc_w) || !read_vector(fc_b)) {
            (*log)(log_time, dbglog::lvl::ERROR, "Failed to LOAD model: Invalid or truncated binary payload.");
            return false;
        }
        (*log)(log_time, dbglog::lvl::info, "Model weights successfully parsed from binary payload.");
        return true;

    } catch (const std::exception& e) {
        // Catch read issues or malformed payloads safely
        std::string err_msg = std::format("Failed to LOAD model: Parse exception -> {}", e.what());
        (*log)(log_time, dbglog::lvl::ERROR, err_msg);
        return false;
    }
}



void DataLoader::shuffle_dataset(std::vector<float>& data_buffer, std::vector<float>& label_buffer, size_t rows, size_t cols, size_t channels) {
    size_t num_images = label_buffer.size();
    size_t img_size = rows * cols * channels;

    if (num_images <= 1) return;

    // Create an index array [0, 1, 2, ... num_images - 1]
    std::vector<size_t> indices(num_images);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle indices using a standard random engine seed
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create temporary buffers to re-arrange data points safely
    std::vector<float> shuffled_data(data_buffer.size());
    std::vector<float> shuffled_labels(label_buffer.size());

    for (size_t i = 0; i < num_images; ++i) {
        size_t old_idx = indices[i];
        
        // Copy label tracking element
        shuffled_labels[i] = label_buffer[old_idx];

        // Copy entire pixel array block matching the single tensor boundary
        std::copy(data_buffer.begin() + (old_idx * img_size),
                  data_buffer.begin() + ((old_idx + 1) * img_size),
                  data_buffer.begin() + (i * img_size));
    }

    // Move data swaps back into reference containers natively
    data_buffer = std::move(shuffled_data);
    label_buffer = std::move(shuffled_labels);
    
    std::println("Dataset buffers shuffled and re-aligned successfully.");
}
// Replaces 'generate_dataset' to pull actual files from disk into your oneDNN flat buffer
bool DataLoader::load_images_from_directory(std::vector<float>& data_buffer, std::vector<float>& label_buffer, const std::string& folder_path, size_t target_count, size_t rows, size_t cols, size_t channels) {
    size_t img_size = channels * rows * cols;
    data_buffer.clear();
    label_buffer.clear();

    if (!fs::exists(folder_path)) {
        std::println(std::cerr, "Error: Directory \"{}\" does not exist!", folder_path);
        return false;
    }

    size_t loaded_count = 0;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (loaded_count >= target_count) break;
        if (!entry.is_regular_file()) continue;

        std::string path = entry.path().string();
        
        // 1. Read image via OpenCV in Grayscale mode (1 channel)
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::println("Skipping invalid image file: {}", path);
            continue; 
        }

        // 2. Normalize spatial geometry to match network constraints (32x24)
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(cols, rows));

        // 3. Convert 8-bit unsigned char pixels [0, 255] to standard floats [0.0f, 1.0f]
        cv::Mat float_img;
        resized_img.convertTo(float_img, CV_32FC1, 1.0f / 255.0f);

        // 4. Push pixel array linearly into the global continuous training buffer
        // For standard grayscale, this cleanly mimics NCHW flat layout spacing
        float* ptr = reinterpret_cast<float*>(float_img.data);
        data_buffer.insert(data_buffer.end(), ptr, ptr + img_size);

        // 5. Deduce labels based on file metadata naming convention (e.g., files containing "human")
        if (path.find("human") != std::string::npos) {
            label_buffer.push_back(1.0f); // Target match
        } else {
            label_buffer.push_back(0.05f); // Background noise floor
        }

        loaded_count++;
    }

    std::println("Successfully imported {} real images into the tensor space.", loaded_count);
    return loaded_count == target_count;
}
std::unordered_map<std::string, float> DataLoader::parse_flir_labels_simdjson(const std::string& json_path) {
    std::unordered_map<std::string, float> image_to_label;
    
    // 1. Create a re-usable DOM parser instances
    dom::parser parser;
    dom::element doc;

    try {
        // 2. Load and parse the JSON file using fallback allocation
        doc = parser.load(json_path);
        
        // FLIR Category ID for 'Person' / 'Human' is typically 1
        constexpr int64_t human_category_id = 1; 
        std::unordered_map<int64_t, std::string> image_id_to_name;

        // 3. Extract information from the "images" array
        dom::array images = doc["images"].get_array();
        for (dom::element img : images) {
            int64_t id = img["id"].get_int64();
            std::string_view file_name = img["file_name"].get_string();
            
            std::string name_str(file_name);
            image_id_to_name[id] = name_str;
            image_to_label[name_str] = 0.05f; // Initialize as empty background noise
        }

        // 4. Extract target annotations from the "annotations" array
        dom::array annotations = doc["annotations"].get_array();
        for (dom::element anno : annotations) {
            int64_t cat_id = anno["category_id"].get_int64();
            if (cat_id == human_category_id) {
                int64_t img_id = anno["image_id"].get_int64();
                
                // If it maps to a known file, mark it as target found (1.0f)
                if (image_id_to_name.contains(img_id)) {
                    image_to_label[image_id_to_name[img_id]] = 1.0f;
                }
            }
        }
        std::println("simdjson parsed FLIR annotations successfully.");
    }
    catch (const simdjson::simdjson_error& e) {
        std::println(std::cerr, "simdjson Parsing Error: {}", e.what());
    }

    return image_to_label;
}

bool DataLoader::load_images_with_map(std::vector<float>& data_buffer, 
                              std::vector<float>& label_buffer,
                              const std::string& folder_path, 
                              const std::unordered_map<std::string, float>& label_map,
                              size_t target_count,
                              size_t rows, size_t cols, size_t channels) {
        
        size_t img_size = channels * rows * cols;
        data_buffer.clear();
        label_buffer.clear();

        if (!fs::exists(folder_path)) {
            std::println(std::cerr, "Error: Dataset directory does not exist: {}", folder_path);
            return false;
        }

        std::vector<fs::directory_entry> image_entries;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) image_entries.push_back(entry);
        }
        available_image_count = image_entries.size();

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::shuffle(image_entries.begin(), image_entries.end(), generator);

        size_t loaded_count = 0;
        for (const auto& entry : image_entries) {
            if (target_count != 0 && loaded_count >= target_count) break;

            std::string filename = entry.path().filename().string();
            std::string full_path = entry.path().string();
            
            // 1. Read the image via OpenCV in Grayscale mode (1 channel for thermal)
            cv::Mat img = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::println("Skipping unreadable image file: {}", filename);
                continue; 
            }

            // 2. Resize image to fit your network input dimensions (e.g., 32x24)
            cv::Mat resized_img;
            cv::resize(img, resized_img, cv::Size(cols, rows));

            // 3. Convert 8-bit unsigned char pixels to normalized 32-bit floats [0.0f, 1.0f]
            cv::Mat float_img;
            resized_img.convertTo(float_img, CV_32FC1, 1.0f / 255.0f);

            // 4. Copy raw pixel data into the continuous flat oneDNN tensor buffer
            float* ptr = reinterpret_cast<float*>(float_img.data);
            data_buffer.insert(data_buffer.end(), ptr, ptr + img_size);

            // 5. Query the label map for this specific filename
            if (label_map.contains(filename)) {
                label_buffer.push_back(label_map.at(filename));
            } else {
                // Default fallback if an image is in the directory but missing from the JSON schema
                label_buffer.push_back(0.05f); 
            }

            loaded_count++;
        }

        if (target_count == 0) {
            std::println("Successfully loaded {} of {} images from FLIR directory.",
                         loaded_count, available_image_count);
            return loaded_count > 0;
        }

        std::println("Randomly loaded {}/{} requested images from a pool of {} FLIR files.",
                     loaded_count, target_count, available_image_count);
        return loaded_count == target_count;
    }

    
std::unordered_map<std::string, float> DataLoader::parse_flir_v2_thermal_labels(const std::string& json_path) {
    std::unordered_map<std::string, float> image_to_label;
    
    dom::parser parser;
    dom::element doc;

    try {
        doc = parser.load(json_path);
        
        // Setup fallbacks for FLIR Category IDs (Defaulting standard person category map)
        int64_t person_category_id = 1; 
        
        // 1. Check if "categories" exists at root, otherwise parse safely
        dom::array categories;
        if (doc["categories"].get_array().error() == SUCCESS) {
            categories = doc["categories"].get_array();
            for (dom::element cat : categories) {
                std::string_view cat_name = cat["name"].get_string();
                if (cat_name == "person" || cat_name == "people") {
                    person_category_id = cat["id"].get_int64();
                    break;
                }
            }
        }

        // 2. Safely capture the Image lookup tracking arrays
        std::unordered_map<int64_t, std::string> image_id_to_name;
        
        if (doc["images"].get_array().error() == SUCCESS) {
            dom::array images = doc["images"].get_array();
            for (dom::element img : images) {
                int64_t id = img["id"].get_int64();
                std::string_view file_name = img["file_name"].get_string();
                
                std::string name_str(file_name);
                image_id_to_name[id] = name_str;
                image_to_label[name_str] = 0.05f; // Noise baseline
            }
        } else if (doc["frames"].get_array().error() == SUCCESS) { 
            // Fallback for native FLIR Conservator structural roots
            dom::array frames = doc["frames"].get_array();
            for (dom::element frame : frames) {
                int64_t id = frame["id"].get_int64();
                std::string_view file_name = frame["file_name"].get_string();
                
                std::string name_str(file_name);
                image_id_to_name[id] = name_str;
                image_to_label[name_str] = 0.05f;
            }
        } else {
            throw std::runtime_error("simdjson schema mismatch: Neither 'images' nor 'frames' field found.");
        }

        // 3. Populate matching target labels into map matrix
        if (doc["annotations"].get_array().error() == SUCCESS) {
            dom::array annotations = doc["annotations"].get_array();
            size_t human_count = 0;
            for (dom::element anno : annotations) {
                int64_t cat_id = anno["category_id"].get_int64();
                if (cat_id == person_category_id) {
                    int64_t img_id = anno["image_id"].get_int64();
                    if (image_id_to_name.contains(img_id)) {
                        image_to_label[image_id_to_name[img_id]] = 1.0f;
                        human_count++;
                    }
                }
            }
            std::println("simdjson parsed FLIR structures successfully. Linked {} targets.", human_count);
        }
    }
    catch (const simdjson_error& e) {
        std::println(std::cerr, "simdjson API Error: {}", e.what());
    }
    catch (const std::exception& e) {
        std::println(std::cerr, "Standard Error checking file fields: {}", e.what());
    }

    return image_to_label;
}
