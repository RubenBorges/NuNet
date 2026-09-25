#include <DenseLayer.hpp>
#include <cmath>    
#include <fstream>  
#include <sstream>  
#include <stdexcept>

using namespace bpy;

DenseLayer::DenseLayer(std::size_t input_size, std::size_t output_size, std::function<double(double)> act_func)
    : weights(input_size, output_size), biases(1, output_size), activation(act_func) {
    
    remanufacture_weights_he(input_size);
}

Matrix DenseLayer::forward(const Matrix& inputs) const {
    // 1. Evaluates the multi-step linear transformation with zero allocations using our optimization hooks
    Matrix output = Matrix::dense(inputs, weights, biases);
    
    // 2. Transform the array values completely in-place
    output.map_inplace(activation);
    
    return output;
}

void DenseLayer::remanufacture_weights(double min, double max) {
    weights.randomize(min, max);
    biases.randomize(min * 0.1, max * 0.1); // Keep biases slightly smaller
}
void DenseLayer::remanufacture_weights_he(std::size_t input_size) {
    // He initialization boundary for uniform distribution: sqrt(6.0 / fan_in)
    double boundary = std::sqrt(6.0 / static_cast<double>(input_size));
    
    // Manufacture weights optimized for ReLU activations
    weights.randomize(-boundary, boundary);
    
    // Biases are traditionally initialized to zero or near-zero for He
    biases.map_inplace([](double) { return 0.01; }); 
}
void DenseLayer::save_to_csv(const std::string& weights_filename, const std::string& biases_filename) const {
    // Export Weights
    std::ofstream w_file(weights_filename);
    if (!w_file.is_open()) throw std::runtime_error("Could not open file to save weights: " + weights_filename);
    
    for (std::size_t r = 0; r < weights.Rows(); ++r) {
        for (std::size_t c = 0; c < weights.Cols(); ++c) {
            w_file << weights(r, c) << (c + 1 == weights.Cols() ? "" : ",");
        }
        w_file << "\n";
    }
    w_file.close();

    // Export Biases
    std::ofstream b_file(biases_filename);
    if (!b_file.is_open()) throw std::runtime_error("Could not open file to save biases: " + biases_filename);
    
    for (std::size_t c = 0; c < biases.Cols(); ++c) {
        b_file << biases(0, c) << (c + 1 == biases.Cols() ? "" : ",");
    }
    b_file << "\n";
    b_file.close();
}

void DenseLayer::load_from_csv(const std::string& weights_filename, const std::string& biases_filename) {
    // Load Weights
    std::ifstream w_file(weights_filename);
    if (!w_file.is_open()) throw std::runtime_error("Could not open file to load weights: " + weights_filename);
    
    std::string line, val;
    std::size_t r = 0;
    while (std::getline(w_file, line) && r < weights.Rows()) {
        std::stringstream ss(line);
        std::size_t c = 0;
        while (std::getline(ss, val, ',') && c < weights.Cols()) {
            weights(r, c) = std::stod(val);
            c++;
        }
        r++;
    }
    w_file.close();

    // Load Biases
    std::ifstream b_file(biases_filename);
    if (!b_file.is_open()) throw std::runtime_error("Could not open file to load biases: " + biases_filename);
    
    if (std::getline(b_file, line)) {
        std::stringstream ss(line);
        std::size_t c = 0;
        while (std::getline(ss, val, ',') && c < biases.Cols()) {
            biases(0, c) = std::stod(val);
            c++;
        }
    }
    b_file.close();
}

const Matrix& DenseLayer::GetWeights() const { return weights; }
const Matrix& DenseLayer::GetBiases() const  { return biases; }


