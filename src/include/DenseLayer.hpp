#pragma once

#include <Matrix.hpp>
#include <functional>
#include <cstddef>
#include <string>
namespace bpy {

class DenseLayer {
    Matrix weights;
    Matrix biases;
    std::function<double(double)> activation;

public:
    // Configures a single neural layer mapping input dimensions to output neurons
    DenseLayer(std::size_t input_size, std::size_t output_size, std::function<double(double)> act_func);
    ~DenseLayer() = default;

    // Evaluates a layer forward pass: Activation((Inputs * Weights) + Biases)
    Matrix forward(const Matrix& inputs) const;
    
    //Tuning
    void remanufacture_weights(double min = -1.0, double max = 1.0);
    void remanufacture_weights_he(std::size_t input_size);

    //Persistence
    void save_to_csv(const std::string& weights_filename, const std::string& biases_filename) const;
    void load_from_csv(const std::string& weights_filename, const std::string& biases_filename);

    // Optional utility getters to view inner layer metrics or states
    const Matrix& GetWeights() const;
    const Matrix& GetBiases() const;
};

} // namespace bpy
