#pragma once

#include <Conv2D.hpp>
#include <DenseLayer.hpp>
#include <Tensor.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace bpy {

class CNN {
public:
    struct Config {
        std::size_t input_channels{1};

        std::size_t filters{8};
        std::size_t kernel_size{3};
        std::size_t convolution_stride{1};
        Padding padding{Padding::Valid};

        std::size_t pool_size{2};
        std::size_t pool_stride{2};

        std::size_t hidden_size{32};
    };

    CNN(
        Config config,
        std::size_t input_rows,
        std::size_t input_cols);

    [[nodiscard]]
    double predict(const Tensor3D& input) const;

    [[nodiscard]]
    Tensor3D convolution_features(
        const Tensor3D& input) const;

    [[nodiscard]]
    Tensor3D pooled_features(
        const Tensor3D& input) const;

    void randomize();

    void save(
        const std::filesystem::path& filename) const;

    void load(
        const std::filesystem::path& filename);

    static CNN load_model(
        const std::filesystem::path& filename);

    [[nodiscard]]
    const Config& configuration() const noexcept {
        return config_;
    }

    [[nodiscard]]
    std::size_t InputRows() const noexcept {
        return input_rows_;
    }

    [[nodiscard]]
    std::size_t InputCols() const noexcept {
        return input_cols_;
    }

    [[nodiscard]]
    std::size_t FlattenedSize() const noexcept {
        return flattened_size_;
    }

    [[nodiscard]]
    const Conv2D& convolution() const noexcept {
        return convolution_;
    }

private:
    Config config_;

    std::size_t input_rows_{};
    std::size_t input_cols_{};
    std::size_t flattened_size_{};

    Conv2D convolution_;

    DenseLayer hidden_;
    DenseLayer output_;

    [[nodiscard]]
    static double relu(double value) noexcept;

    [[nodiscard]]
    static double sigmoid(double value) noexcept;

    [[nodiscard]]
    std::size_t calculate_flattened_size() const;

    void validate_input(
        const Tensor3D& input) const;
};

} // namespace bpy
