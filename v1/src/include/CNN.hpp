#pragma once

#include <Conv2D.hpp>
#include <DenseLayer.hpp>
#include <Tensor.hpp>

#include <cstddef>

namespace bpy {

class CNN {
private:
    Conv2D convolution_;

    std::size_t pool_size_;
    std::size_t pool_stride_;

    DenseLayer hidden_;
    DenseLayer output_;

    [[nodiscard]]
    Tensor3D max_pool(
        const Tensor3D& input) const;

    [[nodiscard]]
    Matrix flatten(
        const Tensor3D& input) const;

    [[nodiscard]]
    static double relu(double x) noexcept;

    [[nodiscard]]
    static double sigmoid(double x) noexcept;

public:
    CNN(
        Conv2D convolution,
        std::size_t flattened_size,
        std::size_t hidden_neurons,
        std::size_t pool_size = 2,
        std::size_t pool_stride = 2);

    ~CNN() = default;

    [[nodiscard]]
    double predict(
        const Tensor3D& input) const;

    [[nodiscard]]
    std::size_t FlattenedSize() const noexcept {
        return hidden_.InputSize();
    }

    [[nodiscard]]
    const Conv2D& Convolution() const noexcept {
        return convolution_;
    }

    [[nodiscard]]
    const DenseLayer& HiddenLayer() const noexcept {
        return hidden_;
    }

    [[nodiscard]]
    const DenseLayer& OutputLayer() const noexcept {
        return output_;
    }
};

} // namespace bpy
