#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/parallel/backend/parallel_for.tbb.hpp"
#include <opencv2/core/parallel/parallel_backend.hpp>

#include <oneapi/tbb/global_control.h>
#include <tbb/global_control.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string_view>
#include <print>
#include <vector>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <DataLoader.hpp>
#include <debug.hpp>

using namespace dnnl;
using namespace bpy;
using uint = unsigned int;
constexpr uint const sampleCount {10};
constexpr uint const epochCount {10};

// Generates a synthetic dataset of mixed images (some with humans, some without)
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void image_to_features(const convolution_forward& convolution, const eltwise_forward& activation,const pooling_forward& pooling,stream& execution_stream,
    memory& images,memory& convolution_weights,memory& convolution_biases,memory& convolution_output,memory& features,memory& pooling_workspace){
        convolution.execute(execution_stream, {
        {DNNL_ARG_SRC, images},
        {DNNL_ARG_WEIGHTS, convolution_weights},
        {DNNL_ARG_BIAS, convolution_biases},
        {DNNL_ARG_DST, convolution_output}
    });
    activation.execute(execution_stream, {
        {DNNL_ARG_SRC, convolution_output},
        {DNNL_ARG_DST, convolution_output}
    });
    pooling.execute(execution_stream, {
        {DNNL_ARG_SRC, convolution_output},
        {DNNL_ARG_DST, features},
        {DNNL_ARG_WORKSPACE, pooling_workspace}
    });
}

std::vector<float> expert_prediction(
    const inner_product_forward& expert,
    stream& execution_stream,
    memory& features,
    memory& weights,
    memory& biases,
    memory& logits,
    size_t batch_count)
{
    expert.execute(execution_stream, {
        {DNNL_ARG_SRC, features},
        {DNNL_ARG_WEIGHTS, weights},
        {DNNL_ARG_BIAS, biases},
        {DNNL_ARG_DST, logits}
    });
    execution_stream.wait();

    const float* values = static_cast<const float*>(logits.get_data_handle());
    std::vector<float> probabilities(batch_count);
    for (size_t sample = 0; sample < batch_count; ++sample) {
        probabilities[sample] = sigmoid(values[sample]);
    }
    return probabilities;
}

std::vector<float> route_predictions(
    const std::vector<std::vector<float>>& expert_probabilities,
    const std::vector<std::vector<float>>& routing_weights)
{
    if (expert_probabilities.empty() || routing_weights.empty()) {
        throw std::invalid_argument("Routing requires expert predictions and gate weights.");
    }

    const size_t expert_count = expert_probabilities.size();
    const size_t batch_count = routing_weights.size();
    for (const auto& predictions : expert_probabilities) {
        if (predictions.size() != batch_count) {
            throw std::invalid_argument("Expert predictions must have the same batch size.");
        }
    }

    std::vector<float> routed(batch_count, 0.0f);
    for (size_t sample = 0; sample < batch_count; ++sample) {
        if (routing_weights[sample].size() != expert_count) {
            throw std::invalid_argument("Each sample needs one routing weight per expert.");
        }

        float weight_sum = 0.0f;
        for (float weight : routing_weights[sample]) {
            if (!std::isfinite(weight) || weight < 0.0f) {
                throw std::invalid_argument("Routing weights must be finite and non-negative.");
            }
            weight_sum += weight;
        }
        if (weight_sum <= 0.0f) {
            throw std::invalid_argument("Routing weights for each sample must sum to a positive value.");
        }

        for (size_t expert_index = 0; expert_index < expert_count; ++expert_index) {
            routed[sample] += expert_probabilities[expert_index][sample] *
                routing_weights[sample][expert_index] / weight_sum;
        }
    }
    return routed;
}


int main(int argc, char* argv[]) {
    cv::parallel::setParallelForBackend(std::make_shared<cv::parallel::tbb::ParallelForBackend>());
    bool force_training = false;
    if (argc == 2 && std::string_view(argv[1]) == "--train") {
        force_training = true;
    } else if (argc != 1) {
        std::println(std::cerr, "Usage: {} [--train]", argv[0]);
        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // TIME & LOGGER INITIALIZATION
    // --------------------------------------------------------
    auto now = std::chrono::system_clock::now();   
    auto current_day = std::chrono::floor<std::chrono::days>(now);
    std::chrono::year_month_day date{current_day}; 
    std::chrono::hh_mm_ss<std::chrono::nanoseconds> time{now - current_day};

    const std::filesystem::path runtime_dir{std::filesystem::absolute(argv[0]).lexically_normal().parent_path()};
    const std::filesystem::path model_dir = runtime_dir.parent_path().string()+="/resources/model";
    dbglog logger(date, runtime_dir / "debug.log");
    DataLoader im(logger);
    
    std::println("{}", cv::getBuildInformation());
    
    tbb::global_control cache_window(tbb::global_control::max_allowed_parallelism, 4);
    cv::setNumThreads(4);


    const std::string model_path = (model_dir / "human_detector.nn").string();
    std::println("Model Located: {}", model_path);
    // --------------------------------------------------------
    // DATASET & BATCH HYPERPARAMETERS
    // --------------------------------------------------------
    constexpr int64_t batch_size = sampleCount; 
    size_t total_dataset_size = 0;
    size_t batch_count = 0;
    
    constexpr int64_t input_rows = 32; constexpr int64_t input_cols = 24; constexpr int64_t input_channels = 1;
    constexpr int64_t filters = 8; constexpr int64_t kernel_size = 3; constexpr int64_t hidden_size = 1; 
    constexpr int64_t pool_size = 2; constexpr int64_t pool_stride = 2;
    constexpr float learning_rate = 0.02f;
    constexpr int epochs = epochCount;

    constexpr int64_t conv_rows = (input_rows - kernel_size) / 1 + 1;
    constexpr int64_t conv_cols = (input_cols - kernel_size) / 1 + 1;
    constexpr int64_t pooled_rows = (conv_rows - pool_size) / pool_stride + 1;
    constexpr int64_t pooled_cols = (conv_cols - pool_size) / pool_stride + 1;
    constexpr int64_t flattened_size = filters * pooled_rows * pooled_cols;

    // --------------------------------------------------------
    // MODEL VARIABLES & CONDITIONAL SYSTEM LOADING
    // --------------------------------------------------------
    std::vector<float> conv_w, conv_b, fc_w, fc_b;
    bool must_train_model = force_training;

    // Attempting load cycle sequence execution
    if(im.load_onednn_model(model_path, conv_w, conv_b, fc_w, fc_b, time) == false) {
        std::println("Pre-trained weight architecture absent. Initializing allocation buffers...");
        must_train_model = true;

        // Allocate fallback baseline weights from scratch
        conv_w.assign(filters * input_channels * kernel_size * kernel_size, 0.05f);
        conv_b.assign(filters, 0.01f);
        fc_w.assign(hidden_size * flattened_size, 0.01f);
        fc_b.assign(hidden_size, 0.0f);
    }

    // --------------------------------------------------------
    // ONEDNN ENGINE & MEMORY DESCRIPTORS
    // --------------------------------------------------------
    engine eng(engine::kind::cpu, 0);
    stream strm(eng);
    std::println("Device Stream Engine Initialized");

    auto conv_src_md = memory::desc({batch_size, input_channels, input_rows, input_cols}, memory::data_type::f32, memory::format_tag::nchw);
    auto conv_weights_md = memory::desc({filters, input_channels, kernel_size, kernel_size}, memory::data_type::f32, memory::format_tag::oihw);
    auto conv_bias_md = memory::desc({filters}, memory::data_type::f32, memory::format_tag::x);
    auto conv_dst_md = memory::desc({batch_size, filters, conv_rows, conv_cols}, memory::data_type::f32, memory::format_tag::nchw);
    auto pool_dst_md = memory::desc({batch_size, filters, pooled_rows, pooled_cols}, memory::data_type::f32, memory::format_tag::nchw);
    auto fc_src_md = memory::desc({batch_size, flattened_size}, memory::data_type::f32, memory::format_tag::nc);
    auto fc_weights_md = memory::desc({hidden_size, flattened_size}, memory::data_type::f32, memory::format_tag::oi);
    auto fc_bias_md = memory::desc({hidden_size}, memory::data_type::f32, memory::format_tag::x);
    auto fc_dst_md = memory::desc({batch_size, hidden_size}, memory::data_type::f32, memory::format_tag::nc);

    // --------------------------------------------------------
    // PRIMITIVE DESCRIPTORS
    // --------------------------------------------------------
    auto conv_fwd_pd = convolution_forward::primitive_desc(eng, prop_kind::forward_training, algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, {1, 1}, {0, 0}, {0, 0});
    auto relu_fwd_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward_training, algorithm::eltwise_relu, conv_dst_md, conv_dst_md, 0.0f, 0.0f);
    auto pool_fwd_pd = pooling_forward::primitive_desc(eng, prop_kind::forward_training, algorithm::pooling_max, conv_dst_md, pool_dst_md, {pool_stride, pool_stride}, {pool_size, pool_size}, {0, 0}, {0, 0}, {0, 0});
    auto fc_fwd_pd = inner_product_forward::primitive_desc(eng, prop_kind::forward_training, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md);
    auto fc_bwd_w_pd = inner_product_backward_weights::primitive_desc(eng, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md, fc_fwd_pd);
    auto fc_bwd_d_pd = inner_product_backward_data::primitive_desc(eng, fc_src_md, fc_weights_md, fc_dst_md, fc_fwd_pd);
    auto pool_bwd_pd = pooling_backward::primitive_desc(eng, algorithm::pooling_max, conv_dst_md, pool_dst_md, {pool_stride, pool_stride}, {pool_size, pool_size}, {0, 0}, {0, 0}, {0, 0}, pool_fwd_pd);
    auto relu_bwd_pd = eltwise_backward::primitive_desc(eng, algorithm::eltwise_relu, conv_dst_md, conv_dst_md, conv_dst_md, 0.0f, 0.0f, relu_fwd_pd);
    auto conv_bwd_w_pd = convolution_backward_weights::primitive_desc(eng, algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, {1, 1}, {0, 0}, {0, 0}, conv_fwd_pd);

    // --------------------------------------------------------
    // DATASET WORKING BUFFERS
    // --------------------------------------------------------
    std::vector<float> dataset_images;
    std::vector<float> dataset_labels;
    const std::string data_path = (model_dir / "FLIR/images_thermal_train/data").string();
    const std::string json_annotation_path = (model_dir / "FLIR/images_thermal_train/coco.json").string();
    
    std::println("Data Path:{} \n Json Annotation Path: {}", data_path, json_annotation_path);

    // Only configure files and run dataset optimization loops if training is forced
    if (must_train_model) {
    try {
        std::filesystem::path cpp_data_dir = model_dir / "FLIR" / "cpp_training_data";
        std::string img_bin = (cpp_data_dir / "images.bin").string();
        std::string lbl_bin = (cpp_data_dir / "labels.bin").string();

        std::println("Loading streamlined tensor blocks directly from binary files...");
        bool load_success = im.load_preprocessed_binaries(dataset_images, dataset_labels, img_bin, lbl_bin);

        if (!load_success) {
            std::println(std::cerr, "Error: Could not load FLIR training binaries from {}", cpp_data_dir.string());
            return EXIT_FAILURE;
        }

        const size_t values_per_image = static_cast<size_t>(input_rows * input_cols * input_channels);
        if (dataset_labels.empty() || dataset_images.size() != dataset_labels.size() * values_per_image) {
            std::println(std::cerr,
                         "Error: FLIR binary sizes do not match the configured {}x{}x{} input and label count.",
                         input_rows, input_cols, input_channels);
            return EXIT_FAILURE;
        }
        
        // Shuffle the arrays so gradients optimize evenly
        im.shuffle_dataset(dataset_images, dataset_labels, input_rows, input_cols, input_channels);
    } 
    catch (const std::exception& e) {
        std::println(std::cerr, "Error: [Data Load Failure] -> {}", e.what());
        return EXIT_FAILURE;
    }
    
    total_dataset_size = dataset_labels.size();
    batch_count = (total_dataset_size + static_cast<size_t>(batch_size) - 1) / static_cast<size_t>(batch_size);
 
        // --------------------------------------------------------
        // TRAINING BACKPROPAGATION GRADIENT STACKS
        // --------------------------------------------------------
        std::vector<float> diff_conv_w(conv_w.size(), 0.0f);
        std::vector<float> diff_conv_b(conv_b.size(), 0.0f);
        std::vector<float> diff_fc_w(fc_w.size(), 0.0f);
        std::vector<float> diff_fc_b(fc_b.size(), 0.0f);

        auto src_mem = memory(conv_src_md, eng);
        auto conv_w_mem = memory(conv_weights_md, eng, conv_w.data());
        auto conv_b_mem = memory(conv_bias_md, eng, conv_b.data());
        auto fc_w_mem = memory(fc_weights_md, eng, fc_w.data());
        auto fc_b_mem = memory(fc_bias_md, eng, fc_b.data());

        auto diff_conv_w_mem = memory(conv_weights_md, eng, diff_conv_w.data());
        auto diff_conv_b_mem = memory(conv_bias_md, eng, diff_conv_b.data());
        auto diff_fc_w_mem = memory(fc_weights_md, eng, diff_fc_w.data());
        auto diff_fc_b_mem = memory(fc_bias_md, eng, diff_fc_b.data());

        auto conv_dst_mem = memory(conv_dst_md, eng);
        auto pool_dst_mem = memory(pool_dst_md, eng);
        auto fc_dst_mem = memory(fc_dst_md, eng);
        auto pool_workspace_mem = memory(pool_fwd_pd.workspace_desc(), eng);

        auto diff_fc_dst_mem = memory(fc_dst_md, eng);
        auto diff_pool_dst_mem = memory(pool_dst_md, eng);
        auto diff_conv_dst_mem = memory(conv_dst_md, eng);

        // Primitives mapping
        auto prim_conv_fwd = convolution_forward(conv_fwd_pd);
        auto prim_relu_fwd = eltwise_forward(relu_fwd_pd);
        auto prim_pool_fwd = pooling_forward(pool_fwd_pd);
        auto prim_fc_fwd = inner_product_forward(fc_fwd_pd);
        auto prim_fc_bwd_w = inner_product_backward_weights(fc_bwd_w_pd);
        auto prim_fc_bwd_d = inner_product_backward_data(fc_bwd_d_pd);
        auto prim_pool_bwd = pooling_backward(pool_bwd_pd);
        auto prim_relu_bwd = eltwise_backward(relu_bwd_pd);
        auto prim_conv_bwd_w = convolution_backward_weights(conv_bwd_w_pd);

        size_t image_size_bytes = input_channels * input_rows * input_cols;

        std::println("Dataset selection: {} of {} available images, Batch Size = {}, Batch Count = {}",total_dataset_size, im.available_image_count, batch_size, batch_count);
        std::println("Starting dataset mini-batch optimization loop...");
        std::println("--------------------------------------------------");

        // --------------------------------------------------------
        // MULTI-BATCH TRAINING LOOP
        // --------------------------------------------------------
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float epoch_total_loss = 0.0f;
            for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
                const size_t offset = batch_index * static_cast<size_t>(batch_size);
                const size_t current_batch_size = std::min(
                    static_cast<size_t>(batch_size), total_dataset_size - offset);

                float* src_handle = static_cast<float*>(src_mem.get_data_handle());
                std::fill_n(src_handle, static_cast<size_t>(batch_size) * image_size_bytes, 0.0f);
                std::copy_n(dataset_images.begin() + (offset * image_size_bytes),
                            current_batch_size * image_size_bytes, src_handle);

                image_to_features(
                    prim_conv_fwd, prim_relu_fwd, prim_pool_fwd, strm,
                    src_mem, conv_w_mem, conv_b_mem, conv_dst_mem,
                    pool_dst_mem, pool_workspace_mem);
                auto expert_probabilities {expert_prediction(
                    prim_fc_fwd, strm, pool_dst_mem, fc_w_mem, fc_b_mem,
                    fc_dst_mem, static_cast<size_t>(batch_size))};
                std::vector<std::vector<float>> expert_outputs{std::move(expert_probabilities)};
                std::vector<std::vector<float>> routing_weights(static_cast<size_t>(batch_size), std::vector<float>{1.0f});
                const auto routed_probabilities = route_predictions(expert_outputs, routing_weights);

                // 2. MINI-BATCH LOSS & LOSS GRADIENT EVALUATION
                float batch_loss = 0.0f;
                float* diff_fc_dst = static_cast<float*>(diff_fc_dst_mem.get_data_handle());

                for (int64_t b = 0; b < batch_size; ++b) {
                    if (static_cast<size_t>(b) >= current_batch_size) {
                        diff_fc_dst[b] = 0.0f;
                        continue;
                    }

                    float probability = routed_probabilities[static_cast<size_t>(b)];
                    float target = dataset_labels[offset + b];

                    batch_loss -= (target * std::log(probability + 1e-7f) + (1.0f - target) * std::log(1.0f - probability + 1e-7f));
                    diff_fc_dst[b] = (probability - target) / static_cast<float>(current_batch_size);
                }
                epoch_total_loss += batch_loss;

                // 3. BACKWARD PROPAGATION STEP
                prim_fc_bwd_w.execute(strm, {{DNNL_ARG_SRC, pool_dst_mem}, {DNNL_ARG_DIFF_DST, diff_fc_dst_mem}, {DNNL_ARG_DIFF_WEIGHTS, diff_fc_w_mem}, {DNNL_ARG_DIFF_BIAS, diff_fc_b_mem}});
                prim_fc_bwd_d.execute(strm, {{DNNL_ARG_DIFF_DST, diff_fc_dst_mem}, {DNNL_ARG_WEIGHTS, fc_w_mem}, {DNNL_ARG_DIFF_SRC, diff_pool_dst_mem}});
                prim_pool_bwd.execute(strm, {{DNNL_ARG_DIFF_DST, diff_pool_dst_mem}, {DNNL_ARG_WORKSPACE, pool_workspace_mem}, {DNNL_ARG_DIFF_SRC, diff_conv_dst_mem}});
                prim_relu_bwd.execute(strm, {{DNNL_ARG_SRC, conv_dst_mem}, {DNNL_ARG_DIFF_DST, diff_conv_dst_mem}, {DNNL_ARG_DIFF_SRC, diff_conv_dst_mem}});
                prim_conv_bwd_w.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DIFF_DST, diff_conv_dst_mem}, {DNNL_ARG_DIFF_WEIGHTS, diff_conv_w_mem}, {DNNL_ARG_DIFF_BIAS, diff_conv_b_mem}});
                strm.wait();

                // 4. SGD OPTIMIZER UPDATES
                for (size_t i = 0; i < fc_w.size(); ++i)   fc_w[i] -= learning_rate * diff_fc_w[i];
                for (size_t i = 0; i < fc_b.size(); ++i)   fc_b[i] -= learning_rate * diff_fc_b[i];
                for (size_t i = 0; i < conv_w.size(); ++i) conv_w[i] -= learning_rate * diff_conv_w[i];
                for (size_t i = 0; i < conv_b.size(); ++i) conv_b[i] -= learning_rate * diff_conv_b[i];
            }

            std::println("Epoch {:02d}/{} completed | Average BCE Loss: {:.6f}",
                         epoch, epochs, epoch_total_loss / static_cast<float>(total_dataset_size));
        }

        std::println("--------------------------------------------------");
        std::println("Batch dataset training complete.");

        // Serialize and log save events exclusively at loop finalization
        im.save_onednn_model(model_path, conv_w, conv_b, fc_w, fc_b);
    } 
    else {
        std::println("--------------------------------------------------");
        std::println("Loaded pre-compiled weights. Preparing FLIR samples for inference...");
        try {
            auto flir_label_map = im.parse_flir_v2_thermal_labels(json_annotation_path);
            if (flir_label_map.empty() || !im.load_images_with_map(dataset_images, dataset_labels, data_path, flir_label_map,static_cast<size_t>(batch_size), input_rows, input_cols, input_channels)) {
                std::println(std::cerr, "Error: Could not load FLIR samples for inference.");
                return EXIT_FAILURE;
            }
            total_dataset_size = dataset_labels.size();
            batch_count = (total_dataset_size + static_cast<size_t>(batch_size) - 1) / static_cast<size_t>(batch_size);
            std::println("Dataset selection: {} of {} available images, Batch Size = {}, Batch Count = {}",
                         total_dataset_size, im.available_image_count, batch_size, batch_count);
        } catch (const std::exception& e) {
            std::println(std::cerr, "Error: Failed to prepare inference samples: {}", e.what());
            return EXIT_FAILURE;
        }
    }

    const size_t expected_conv_weights = static_cast<size_t>(filters * input_channels * kernel_size * kernel_size);
    const size_t expected_conv_biases = static_cast<size_t>(filters);
    const size_t expected_fc_weights = static_cast<size_t>(hidden_size * flattened_size);
    const size_t expected_fc_biases = static_cast<size_t>(hidden_size);
    if (conv_w.size() != expected_conv_weights || conv_b.size() != expected_conv_biases ||
        fc_w.size() != expected_fc_weights || fc_b.size() != expected_fc_biases) {
        std::println(std::cerr, "Error: Model parameter sizes do not match the configured network architecture.");
        return EXIT_FAILURE;
    }

    const size_t inference_image_values = static_cast<size_t>(input_channels * input_rows * input_cols);
    if (dataset_images.size() < total_dataset_size * inference_image_values ||
        dataset_labels.size() < total_dataset_size) {
        std::println(std::cerr, "Error: Not enough FLIR samples were loaded for inference.");
        return EXIT_FAILURE;
    }

    auto inference_src_mem = memory(conv_src_md, eng);
    auto inference_conv_weights = memory(conv_weights_md, eng, conv_w.data());
    auto inference_conv_biases = memory(conv_bias_md, eng, conv_b.data());
    auto inference_fc_weights = memory(fc_weights_md, eng, fc_w.data());
    auto inference_fc_biases = memory(fc_bias_md, eng, fc_b.data());
    auto inference_conv_dst = memory(conv_dst_md, eng);
    auto inference_pool_dst = memory(pool_dst_md, eng);
    auto inference_fc_src = memory(fc_src_md, eng, inference_pool_dst.get_data_handle());
    auto inference_fc_dst = memory(fc_dst_md, eng);
    auto inference_pool_workspace = memory(pool_fwd_pd.workspace_desc(), eng);
    auto inference_conv_primitive = convolution_forward(conv_fwd_pd);
    auto inference_relu_primitive = eltwise_forward(relu_fwd_pd);
    auto inference_pool_primitive = pooling_forward(pool_fwd_pd);
    auto inference_expert_primitive = inner_product_forward(fc_fwd_pd);

    size_t human_prediction_count = 0;
    size_t displayed_prediction_count = 0;
    constexpr size_t max_displayed_predictions = 25;
    std::println("Inference results (showing up to {} samples):", max_displayed_predictions);
    for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        const size_t offset = batch_index * static_cast<size_t>(batch_size);
        const size_t current_batch_size = std::min(
            static_cast<size_t>(batch_size), total_dataset_size - offset);
        const size_t current_batch_values = current_batch_size * inference_image_values;
        float* inference_source = static_cast<float*>(inference_src_mem.get_data_handle());
        std::fill_n(inference_source, static_cast<size_t>(batch_size) * inference_image_values, 0.0f);
        std::copy_n(dataset_images.begin() + offset * inference_image_values,
                    current_batch_values, inference_source);

        image_to_features(
            inference_conv_primitive, inference_relu_primitive, inference_pool_primitive, strm,
            inference_src_mem, inference_conv_weights, inference_conv_biases,
            inference_conv_dst, inference_pool_dst, inference_pool_workspace);
        auto expert_probabilities = expert_prediction(
            inference_expert_primitive, strm, inference_fc_src, inference_fc_weights,
            inference_fc_biases, inference_fc_dst, static_cast<size_t>(batch_size));
        std::vector<std::vector<float>> expert_outputs{std::move(expert_probabilities)};
        std::vector<std::vector<float>> routing_weights(
            static_cast<size_t>(batch_size), std::vector<float>{1.0f});
        const auto routed_probabilities = route_predictions(expert_outputs, routing_weights);

        for (size_t sample = 0; sample < current_batch_size; ++sample) {
            const float confidence = routed_probabilities[sample];
            if (confidence >= 0.5f) ++human_prediction_count;
            if (displayed_prediction_count < max_displayed_predictions) {
                std::println("  Sample {}: {} (confidence {:.4f}, label {:.2f})",
                             offset + sample + 1, confidence >= 0.5f ? "human" : "no human",
                             confidence, dataset_labels[offset + sample]);
                ++displayed_prediction_count;
            }
        }
    }
    std::println("Inference completed for {} sampled images from {} available: {} human, {} no human.",
                 total_dataset_size, im.available_image_count, human_prediction_count,
                 total_dataset_size - human_prediction_count);

    return 0;
}
