#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

using namespace dnnl;

int main() {
    try {
        // 1. Initialize Engine and Stream (Targeting CPU)
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        // 2. Define Dimensions
        // Input: 1 batch, 1 channel (grayscale), 4x4 image
        memory::dims src_dims = {1, 1, 4, 4};
        // Weights: 1 output channel, 1 input channel, 3x3 kernel
        memory::dims weights_dims = {1, 1, 3, 3};
        // Bias: 1 value per output channel
        memory::dims bias_dims = {1};
        // Output: 1 batch, 1 channel, 2x2 image (with valid padding and 1 stride)
        memory::dims dst_dims = {1, 1, 2, 2};

        memory::dims strides = {1, 1};
        memory::dims padding = {0, 0};

        // 3. Define Memory Descriptors
        auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
        auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

        // 4. Source/Input Data Definition
        std::vector<float> src_data = {
            1.0f, 2.0f, 3.0f, 0.0f,
            0.0f, 1.0f, 2.0f, 3.0f,
            3.0f, 0.0f, 1.0f, 2.0f,
            2.0f, 3.0f, 0.0f, 1.0f
        };
        // Weights definition (Vertical Edge Detector)
        std::vector<float> weights_data = {
            1.0f, 0.0f, -1.0f,
            1.0f, 0.0f, -1.0f,
            1.0f, 0.0f, -1.0f
        };
        std::vector<float> bias_data = {0.5f};

        // Output buffers
        std::vector<float> conv_dst_data(4, 0.0f);
        std::vector<float> relu_dst_data(4, 0.0f);

        // Wrap data into oneDNN memory objects
        auto src_mem = memory(src_md, eng, src_data.data());
        auto weights_mem = memory(weights_md, eng, weights_data.data());
        auto bias_mem = memory(bias_md, eng, bias_data.data());
        auto conv_dst_mem = memory(dst_md, eng, conv_dst_data.data());
        auto relu_dst_mem = memory(dst_md, eng, relu_dst_data.data());

        // 5. Convolution Primitive Configuration
        auto conv_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md,
            weights_md,
            bias_md,
            dst_md,
            strides,
            padding,
            padding
            );
        auto conv_prim = convolution_forward(conv_pd);

        // 6. ReLU Primitive Configuration
        auto relu_pd = eltwise_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::eltwise_relu,
            dst_md,
            dst_md,
            0.0f
            );
        auto relu_prim = eltwise_forward(relu_pd);

        // 7. Execution Pipeline
        // Step A: Run Convolution
        conv_prim.execute(s, {
                                 {DNNL_ARG_SRC, src_mem},
                                 {DNNL_ARG_WEIGHTS, weights_mem},
                                 {DNNL_ARG_BIAS, bias_mem},
                                 {DNNL_ARG_DST, conv_dst_mem}
                             });

        // Step B: Run ReLU on Convolution Output
        relu_prim.execute(s, {
                                 {DNNL_ARG_SRC, conv_dst_mem},
                                 {DNNL_ARG_DST, relu_dst_mem}
                             });

        // Wait for execution pipelines to finish
        s.wait();

        // 8. Print Results
        std::cout << "--- Mini-CNN Layer Output ---" << std::endl;
        std::cout << "Input Image (4x4):" << std::endl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) std::cout << src_data[i * 4 + j] << " ";
            std::cout << std::endl;
        }

        std::cout << "\\nConvolution Output (with 0.5 bias):" << std::endl;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) std::cout << conv_dst_data[i * 2 + j] << " ";
            std::cout << std::endl;
        }

        std::cout << "\\nFinal ReLU Output (2x2):" << std::endl;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) std::cout << relu_dst_data[i * 2 + j] << " ";
            std::cout << std::endl;
        }

    } catch (error& e) {
        std::cerr << "oneDNN error caught: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
