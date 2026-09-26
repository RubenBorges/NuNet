#pragma once

#include <Tensor.hpp>

#include <cstddef>
#include <random>

namespace bpy {

enum class Padding {
    Valid,
    Same
};

class Conv2D {
private:
    std::size_t input_channels_;
    std::size_t output_channels_;
    std::size_t kernel_size_;
    std::size_t stride_;
    Padding padding_;

    // [output_channel][input_channel][kernel][kernel]
    Tensor3D weights_;

    // [output_channel]
    std::vector<double> biases_;

    [[nodiscard]]
    std::size_t output_rows(std::size_t input_rows) const noexcept;

    [[nodiscard]]
    std::size_t output_cols(std::size_t input_cols) const noexcept;

    [[nodiscard]]
    double weight(
        std::size_t output_channel,
        std::size_t input_channel,
        std::size_t kernel_row,
        std::size_t kernel_col) const noexcept;

public:
    Conv2D(
        std::size_t input_channels,
        std::size_t output_channels,
        std::size_t kernel_size,
        std::size_t stride = 1,
        Padding padding = Padding::Valid);

    ~Conv2D() = default;

    Conv2D(const Conv2D&) = default;
    Conv2D(Conv2D&&) noexcept = default;

    Conv2D& operator=(const Conv2D&) = default;
    Conv2D& operator=(Conv2D&&) noexcept = default;

    [[nodiscard]]
    Tensor3D forward(const Tensor3D& input) const;

    void randomize(
        double min = -0.1,
        double max = 0.1);

    [[nodiscard]]
    std::size_t InputChannels() const noexcept {
        return input_channels_;
    }

    [[nodiscard]]
    std::size_t OutputChannels() const noexcept {
        return output_channels_;
    }

    [[nodiscard]]
    std::size_t KernelSize() const noexcept {
        return kernel_size_;
    }

    [[nodiscard]]
    std::size_t Stride() const noexcept {
        return stride_;
    }

    [[nodiscard]]
    Padding PaddingMode() const noexcept {
        return padding_;
    }

    [[nodiscard]]
    std::size_t OutputRows(
        std::size_t input_rows) const noexcept {
        return output_rows(input_rows);
    }

    [[nodiscard]]
    std::size_t OutputCols(
        std::size_t input_cols) const noexcept {
        return output_cols(input_cols);
    }
};

} // namespace bpy
