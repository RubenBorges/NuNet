#include <Tensor.hpp>

#include <stdexcept>

namespace bpy {

Tensor3D::Tensor3D(
    std::size_t channels,
    std::size_t rows,
    std::size_t cols,
    double init_value)
    : channels_{channels},
      rows_{rows},
      cols_{cols},
      data_(channels * rows * cols, init_value)
{
    if (channels == 0 || rows == 0 || cols == 0)
        throw std::invalid_argument(
            "Tensor dimensions must be greater than zero");
}

double& Tensor3D::operator()(
    std::size_t channel,
    std::size_t row,
    std::size_t col) noexcept
{
    return data_[
        (channel * rows_ + row) * cols_ + col
    ];
}

const double& Tensor3D::operator()(
    std::size_t channel,
    std::size_t row,
    std::size_t col) const noexcept
{
    return data_[
        (channel * rows_ + row) * cols_ + col
    ];
}

} // namespace bpy
