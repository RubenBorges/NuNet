#pragma once

#include <Matrix.hpp>

#include <cstddef>
#include <functional>
#include <string>

namespace bpy {

class DenseLayer {
private:
    Matrix weights_;
    Matrix biases_;

    std::function<double(double)> activation_;

public:
    DenseLayer(
        std::size_t input_size,
        std::size_t output_size,
        std::function<double(double)> activation);

    ~DenseLayer() = default;

    DenseLayer(const DenseLayer&) = default;
    DenseLayer(DenseLayer&&) noexcept = default;

    DenseLayer& operator=(const DenseLayer&) = default;
    DenseLayer& operator=(DenseLayer&&) noexcept = default;

    [[nodiscard]]
    Matrix forward(const Matrix& inputs) const;

    void randomize(
        double min = -1.0,
        double max = 1.0);

    void initialize_he(
        std::size_t input_size);

    void save_to_csv(
        const std::string& weights_filename,
        const std::string& biases_filename) const;

    void load_from_csv(
        const std::string& weights_filename,
        const std::string& biases_filename);

    [[nodiscard]]
    const Matrix& GetWeights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const Matrix& GetBiases() const noexcept {
        return biases_;
    }

    [[nodiscard]]
    std::size_t InputSize() const noexcept {
        return weights_.Rows();
    }

    [[nodiscard]]
    std::size_t OutputSize() const noexcept {
        return weights_.Cols();
    }
};

} // namespace bpy
