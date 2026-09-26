#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <Matrix.hpp>
#include <DenseLayer.hpp>

using DenseLayer = bpy::DenseLayer;
using Matrix = bpy::Matrix; // creates an alias named Matrix

// Activation Functions:
double relu(double x) { return std::max(0.0, x); }
double sigmoid(double x) {return 1.0 / (1.0 + std::exp(-x));}

int main() {
    std::srand(std::time(nullptr));

    // 1. INPUT IMAGE (32x24 pixels, 1 channel)
    const int INPUT_ROWS = 32;
    const int INPUT_COLS = 24;
    Matrix input_image(INPUT_ROWS, INPUT_COLS);
    
    // Simulate a mock human heat signature in the center of the thermal image
    for (auto r = 0; r < INPUT_ROWS; ++r) {
        for (auto c = 0; c < INPUT_COLS; ++c) {
            if (r >= 12 && r <= 22 && c >= 8 && c <= 18) {
                input_image.set(r, c, 0.9); // High temperature pixels
            } else {
                input_image.set(r, c, 0.1); // Background ambient noise
            }
        }
    }
    std::cout << "--- Forward Pass Simulation for 32x24 Thermal Image ---" << std::endl;
    std::cout << "Input Dimensions: " << INPUT_ROWS << "x" << INPUT_COLS << " (1 Channel)" << std::endl;

    // 2. CONVOLUTIONAL LAYER DESIGN
    // 3x3 kernel, Stride = 1, Padding = Valid (No padding)
    const int KERNEL_SIZE = 3;
    const int STRIDE = 1;
    
    Matrix conv_kernel(KERNEL_SIZE, KERNEL_SIZE);
    Matrix::randomize_matrix(conv_kernel);
    double conv_bias = ((double)rand() / RAND_MAX) - 0.5;

    // Calculate valid output dimensions: (Input - Kernel) / Stride + 1
    int conv_rows = (INPUT_ROWS - KERNEL_SIZE) / STRIDE + 1; // (32 - 3) + 1 = 30
    int conv_cols = (INPUT_COLS - KERNEL_SIZE) / STRIDE + 1; // (24 - 3) + 1 = 22
    Matrix conv_output(conv_rows, conv_cols);

    // Perform Convolution + ReLU Activation
    for (auto r = 0; r < conv_rows; ++r) {
        for (auto c = 0; c < conv_cols; ++c) {
            double sum = 0.0;
            // Slide 3x3 kernel over the input patch
            for (auto kr = 0; kr < KERNEL_SIZE; ++kr) {
                for (auto kc = 0; kc < KERNEL_SIZE; ++kc) {
                    sum += input_image.get(r + kr, c + kc) * conv_kernel.get(kr, kc);
                }
            }
            conv_output.set(r, c, relu(sum + conv_bias));
        }
    }
    std::cout << "Conv Layer Output (Valid Padding): " << conv_rows << "x" << conv_cols << std::endl;

    // 3. MAX POOLING LAYER (2x2 pool size, Stride = 2)
    const int POOL_SIZE = 2;
    const int POOL_STRIDE = 2;
    int pool_rows = conv_rows / POOL_STRIDE; // 30 / 2 = 15
    int pool_cols = conv_cols / POOL_STRIDE; // 22 / 2 = 11
    Matrix pool_output(pool_rows, pool_cols);

    for (auto r = 0; r < pool_rows; ++r) {
        for (auto c = 0; c < pool_cols; ++c) {
            double max_val = -999.0;
            // Look at 2x2 local window
            for (auto pr = 0; pr < POOL_SIZE; ++pr) {
                for (auto pc = 0; pc < POOL_SIZE; ++pc) {
                    double val = conv_output.get(r * POOL_STRIDE + pr, c * POOL_STRIDE + pc);
                    if (val > max_val) max_val = val;
                }
            }
            pool_output.set(r, c, max_val);
        }
    }
    std::cout << "MaxPooling Output (2x2 Pool, Stride 2): " << pool_rows << "x" << pool_cols << std::endl;

    // 4. FLATTEN LAYER
    // Convert 2D pooled feature map (16x12) into a 1D vector
    int flattened_size = pool_rows * pool_cols; // 16 * 12 = 192 features
    std::vector<double> flattened_features(flattened_size);
    int idx = 0;
    for (auto r = 0; r < pool_rows; ++r) {
        for (auto c = 0; c < pool_cols; ++c) {
            flattened_features[idx++] = pool_output.get(r, c);
        }
    }
    std::cout << "Flattened Vector Size: " << flattened_size << " dimensions" << std::endl;

    // 5. FULLY CONNECTED LAYER (Dense Layer mapping 192 features to 1 output)
    std::vector<double> fc_weights(flattened_size);
    for (auto i = 0; i < flattened_size; ++i) {
        fc_weights[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    double fc_bias = ((double)rand() / RAND_MAX) - 0.5;

    // Calculate final weighted sum
    double linear_output = 0.0;
    for (auto i = 0; i < flattened_size; ++i) {
        linear_output += flattened_features[i] * fc_weights[i];
    }
    linear_output += fc_bias;

    // Output Layer activation function
    double human_probability = sigmoid(linear_output);

    std::cout << "\nPrediction Result (Probability): " << human_probability << std::endl;
    std::cout << "Conclusion: " << (human_probability >= 0.5 ? "HUMAN DETECTED" : "NO HUMAN DETECTED") << std::endl;

    return 0;
}