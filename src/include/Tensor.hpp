#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace bpy {

class Tensor3D {
private:
    std::size_t channels_{};
    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<double> data_;

    [[nodiscard]]
    std::size_t index(
        std::size_t channel,
        std::size_t row,
        std::size_t col) const noexcept
    {
        return (channel * rows_ + row) * cols_ + col;
    }

public:
    Tensor3D(
        std::size_t channels,
        std::size_t rows,
        std::size_t cols,
        double init_value = 0.0);

    Tensor3D(const Tensor3D&) = default;
    Tensor3D(Tensor3D&&) noexcept = default;
    Tensor3D& operator=(const Tensor3D&) = default;
    Tensor3D& operator=(Tensor3D&&) noexcept = default;
    ~Tensor3D() = default;

    double& operator()(
        std::size_t channel,
        std::size_t row,
        std::size_t col) noexcept;

    const double& operator()(
        std::size_t channel,
        std::size_t row,
        std::size_t col) const noexcept;

    [[nodiscard]]
    std::size_t Channels() const noexcept { return channels_; }

    [[nodiscard]]
    std::size_t Rows() const noexcept { return rows_; }

    [[nodiscard]]
    std::size_t Cols() const noexcept { return cols_; }

    [[nodiscard]]
    std::size_t Size() const noexcept { return data_.size(); }

    [[nodiscard]]
    double* Data() noexcept { return data_.data(); }

    [[nodiscard]]
    const double* Data() const noexcept { return data_.data(); }

    void fill(double value) noexcept;

    void randomize(
        double min = -1.0,
        double max = 1.0);

    [[nodiscard]]
    Tensor3D relu() const;

    [[nodiscard]]
    Tensor3D max_pool(
        std::size_t pool_size,
        std::size_t stride) const;

    [[nodiscard]]
    std::vector<double> flatten() const;

    [[nodiscard]]
    Tensor3D convolve(
        const Tensor3D& kernels,
        const std::vector<double>& biases,
        std::size_t stride = 1,
        std::size_t padding = 0) const;
};

} // namespace bpy
