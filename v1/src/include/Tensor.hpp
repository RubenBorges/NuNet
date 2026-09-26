#pragma once

#include <cstddef>
#include <vector>

namespace bpy {

class Tensor3D {
private:
    std::size_t channels_;
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:
    Tensor3D(
        std::size_t channels,
        std::size_t rows,
        std::size_t cols,
        double init_value = 0.0);

    ~Tensor3D() = default;

    Tensor3D(const Tensor3D&) = default;
    Tensor3D(Tensor3D&&) noexcept = default;

    Tensor3D& operator=(const Tensor3D&) = default;
    Tensor3D& operator=(Tensor3D&&) noexcept = default;

    [[nodiscard]]
    double& operator()(
        std::size_t channel,
        std::size_t row,
        std::size_t col) noexcept;

    [[nodiscard]]
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
    std::size_t Size() const noexcept {
        return data_.size();
    }

    [[nodiscard]]
    const std::vector<double>& Data() const noexcept {
        return data_;
    }

    [[nodiscard]]
    std::vector<double>& Data() noexcept {
        return data_;
    }
};

} // namespace bpy
