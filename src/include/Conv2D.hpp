#pragma once

#include <Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace bpy {

enum class Padding {
    Valid,
    Same
};

class Conv2D {
public:
    Conv2D(
        std::size_t input_channels,
        std::size_t output_channels,
        std::size_t kernel_size,
        std::size_t stride = 1,
        Padding padding = Padding::Valid);

    Conv2D(const Conv2D&) = default;
    Conv2D(Conv2D&&) noexcept = default;
    Conv2D& operator=(const Conv2D&) = default;
    Conv2D& operator=(Conv2D&&) noexcept = default;
    ~Conv2D() = default;

    [[nodiscard]]
    Tensor3D forward(
        const Tensor3D& input,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Sender<Tensor3D> forward_async(
        const Tensor3D& input,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    void randomize();

    void save(std::ostream& stream) const;
    void load(std::istream& stream);

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
    Padding GetPadding() const noexcept {
        return padding_;
    }

    [[nodiscard]]
    const Tensor3D& Weights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const std::vector<double>& Biases() const noexcept {
        return biases_;
    }

private:
    std::size_t input_channels_{};
    std::size_t output_channels_{};
    std::size_t kernel_size_{};
    std::size_t stride_{};
    Padding padding_{Padding::Valid};

    Tensor3D weights_;
    std::vector<double> biases_;
};

} // namespace bpy
