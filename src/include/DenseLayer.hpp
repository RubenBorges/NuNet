#pragma once

#include <Matrix.hpp>

#include <cstddef>
#include <functional>
#include <istream>
#include <ostream>

namespace bpy {

class DenseLayer {
public:
    using Activation = std::function<double(double)>;

    DenseLayer(
        std::size_t input_size,
        std::size_t output_size,
        Activation activation);

    [[nodiscard]]
    Matrix forward(const Matrix& input) const;

    void randomize();

    void save(std::ostream& stream) const;
    void load(std::istream& stream);

    [[nodiscard]]
    std::size_t InputSize() const noexcept {
        return weights_.Rows();
    }

    [[nodiscard]]
    std::size_t OutputSize() const noexcept {
        return weights_.Cols();
    }

    [[nodiscard]]
    const Matrix& Weights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const Matrix& Biases() const noexcept {
        return biases_;
    }

private:
    Matrix weights_;
    Matrix biases_;
    Activation activation_;
};

} // namespace bpy
