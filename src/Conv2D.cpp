#include <Conv2D.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace bpy {

Conv2D::Conv2D(
    std::size_t input_channels,
    std::size_t output_channels,
    std::size_t kernel_size,
    std::size_t stride,
    Padding padding)
    : input_channels_{input_channels},
      output_channels_{output_channels},
      kernel_size_{kernel_size},
      stride_{stride},
      padding_{padding},
      weights_{
          output_channels,
          input_channels,
          kernel_size * kernel_size,
          0.0
      },
      biases_(output_channels, 0.0)
{
    if (input_channels == 0)
        throw std::invalid_argument(
            "Conv2D requires at least one input channel");

    if (output_channels == 0)
        throw std::invalid_argument(
            "Conv2D requires at least one output channel");

    if (kernel_size == 0)
        throw std::invalid_argument(
            "Conv2D kernel size must be greater than zero");

    if (stride == 0)
        throw std::invalid_argument(
            "Conv2D stride must be greater than zero");

    randomize();
}

std::size_t Conv2D::output_rows(
    std::size_t input_rows) const noexcept
{
    if (padding_ == Padding::Same)
        return (input_rows + stride_ - 1) / stride_;

    if (input_rows < kernel_size_)
        return 0;

    return (input_rows - kernel_size_) / stride_ + 1;
}

std::size_t Conv2D::output_cols(
    std::size_t input_cols) const noexcept
{
    if (padding_ == Padding::Same)
        return (input_cols + stride_ - 1) / stride_;

    if (input_cols < kernel_size_)
        return 0;

    return (input_cols - kernel_size_) / stride_ + 1;
}

double Conv2D::weight(
    std::size_t output_channel,
    std::size_t input_channel,
    std::size_t kernel_row,
    std::size_t kernel_col) const noexcept
{
    return weights_(
        output_channel,
        input_channel,
        kernel_row * kernel_size_ + kernel_col
    );
}

void Conv2D::randomize(double min, double max)
{
    if (!(min < max))
        throw std::invalid_argument(
            "Conv2D randomization requires min < max");

    std::random_device rd;
    std::mt19937 generator(rd());

    const double fan_in =
        static_cast<double>(
            input_channels_ *
            kernel_size_ *
            kernel_size_);

    // He-style initialization.
    const double limit =
        std::sqrt(6.0 / fan_in);

    std::uniform_real_distribution<double> distribution(
        -limit,
        limit
    );

    for (double& value : weights_.Data())
        value = distribution(generator);

    std::fill(
        biases_.begin(),
        biases_.end(),
        0.0
    );
}

Tensor3D Conv2D::forward(
    const Tensor3D& input) const
{
    if (input.Channels() != input_channels_)
        throw std::invalid_argument(
            "Conv2D input channel count does not match layer");

    const std::size_t out_rows =
        output_rows(input.Rows());

    const std::size_t out_cols =
        output_cols(input.Cols());

    if (out_rows == 0 || out_cols == 0)
        throw std::invalid_argument(
            "Conv2D kernel is larger than the input");

    Tensor3D output{
        output_channels_,
        out_rows,
        out_cols,
        0.0
    };

    const bool same =
        padding_ == Padding::Same;

    const std::ptrdiff_t pad =
        same
            ? static_cast<std::ptrdiff_t>(kernel_size_ / 2)
            : 0;

    for (std::size_t oc = 0;
         oc < output_channels_;
         ++oc)
    {
        for (std::size_t orow = 0;
             orow < out_rows;
             ++orow)
        {
            for (std::size_t ocol = 0;
                 ocol < out_cols;
                 ++ocol)
            {
                double sum = biases_[oc];

                const auto input_row_start =
                    static_cast<std::ptrdiff_t>(
                        orow * stride_) - pad;

                const auto input_col_start =
                    static_cast<std::ptrdiff_t>(
                        ocol * stride_) - pad;

                for (std::size_t ic = 0;
                     ic < input_channels_;
                     ++ic)
                {
                    for (std::size_t kr = 0;
                         kr < kernel_size_;
                         ++kr)
                    {
                        const auto ir =
                            input_row_start +
                            static_cast<std::ptrdiff_t>(kr);

                        if (ir < 0 ||
                            ir >= static_cast<std::ptrdiff_t>(
                                input.Rows()))
                            continue;

                        for (std::size_t kc = 0;
                             kc < kernel_size_;
                             ++kc)
                        {
                            const auto ic_col =
                                input_col_start +
                                static_cast<std::ptrdiff_t>(kc);

                            if (ic_col < 0 ||
                                ic_col >= static_cast<std::ptrdiff_t>(
                                    input.Cols()))
                                continue;

                            sum +=
                                input(
                                    ic,
                                    static_cast<std::size_t>(ir),
                                    static_cast<std::size_t>(ic_col)
                                )
                                *
                                weight(
                                    oc,
                                    ic,
                                    kr,
                                    kc
                                );
                        }
                    }
                }

                // ReLU.
                output(
                    oc,
                    orow,
                    ocol
                ) = std::max(0.0, sum);
            }
        }
    }

    return output;
}

} // namespace bpy
