#include <Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
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
        throw std::invalid_argument("Tensor dimensions must be greater than zero");
}

double& Tensor3D::operator()(
    std::size_t channel,
    std::size_t row,
    std::size_t col) noexcept
{
    return data_[index(channel, row, col)];
}

const double& Tensor3D::operator()(
    std::size_t channel,
    std::size_t row,
    std::size_t col) const noexcept
{
    return data_[index(channel, row, col)];
}

void Tensor3D::fill(double value) noexcept
{
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor3D::randomize(double min, double max)
{
    if (!(min < max))
        throw std::invalid_argument("Randomization minimum must be less than maximum");

    static thread_local std::mt19937_64 engine{
        std::random_device{}()
    };

    std::uniform_real_distribution<double> distribution(min, max);

    for (double& value : data_)
        value = distribution(engine);
}

Tensor3D Tensor3D::relu() const
{
    Tensor3D result{channels_, rows_, cols_};

    std::transform(
        data_.begin(),
        data_.end(),
        result.data_.begin(),
        [](double value) noexcept {
            return std::max(0.0, value);
        });

    return result;
}

Tensor3D Tensor3D::max_pool(
    std::size_t pool_size,
    std::size_t stride) const
{
    if (pool_size == 0 || stride == 0)
        throw std::invalid_argument("Pool size and stride must be greater than zero");

    if (pool_size > rows_ || pool_size > cols_)
        throw std::invalid_argument("Pool window is larger than tensor");

    const std::size_t output_rows =
        (rows_ - pool_size) / stride + 1;

    const std::size_t output_cols =
        (cols_ - pool_size) / stride + 1;

    Tensor3D result{
        channels_,
        output_rows,
        output_cols
    };

    for (std::size_t ch = 0; ch < channels_; ++ch)
    {
        for (std::size_t r = 0; r < output_rows; ++r)
        {
            const std::size_t input_r = r * stride;

            for (std::size_t c = 0; c < output_cols; ++c)
            {
                const std::size_t input_c = c * stride;

                double maximum =
                    -std::numeric_limits<double>::infinity();

                for (std::size_t pr = 0; pr < pool_size; ++pr)
                {
                    for (std::size_t pc = 0; pc < pool_size; ++pc)
                    {
                        maximum = std::max(
                            maximum,
                            (*this)(
                                ch,
                                input_r + pr,
                                input_c + pc));
                    }
                }

                result(ch, r, c) = maximum;
            }
        }
    }

    return result;
}

std::vector<double> Tensor3D::flatten() const
{
    return data_;
}

Tensor3D Tensor3D::convolve(
    const Tensor3D& kernels,
    const std::vector<double>& biases,
    std::size_t stride,
    std::size_t padding) const
{
    if (stride == 0)
        throw std::invalid_argument("Convolution stride must be greater than zero");

    if (kernels.Rows() != kernels.Cols())
        throw std::invalid_argument("Convolution kernels must be square");

    if (kernels.Channels() == 0)
        throw std::invalid_argument("Convolution requires at least one output channel");

    const std::size_t kernel_size = kernels.Rows();

    if (kernel_size > rows_ + 2 * padding ||
        kernel_size > cols_ + 2 * padding)
    {
        throw std::invalid_argument(
            "Convolution kernel is larger than padded input");
    }

    if (biases.size() != kernels.Channels())
        throw std::invalid_argument(
            "Convolution bias count must match output channels");

    const std::size_t output_rows =
        (rows_ + 2 * padding - kernel_size) / stride + 1;

    const std::size_t output_cols =
        (cols_ + 2 * padding - kernel_size) / stride + 1;

    /*
        kernels layout:

        [output_channel][input_channel][kernel_row][kernel_col]

        flattened into:

        channel = output_channel * input_channels
                + input_channel
    */

    Tensor3D output{
        kernels.Channels(),
        output_rows,
        output_cols
    };

    for (std::size_t out_ch = 0;
         out_ch < kernels.Channels();
         ++out_ch)
    {
        const std::size_t kernel_channel_base =
            out_ch * channels_;

        for (std::size_t out_r = 0;
             out_r < output_rows;
             ++out_r)
        {
            const std::ptrdiff_t base_r =
                static_cast<std::ptrdiff_t>(out_r * stride) -
                static_cast<std::ptrdiff_t>(padding);

            for (std::size_t out_c = 0;
                 out_c < output_cols;
                 ++out_c)
            {
                const std::ptrdiff_t base_c =
                    static_cast<std::ptrdiff_t>(out_c * stride) -
                    static_cast<std::ptrdiff_t>(padding);

                double sum = biases[out_ch];

                for (std::size_t in_ch = 0;
                     in_ch < channels_;
                     ++in_ch)
                {
                    const std::size_t kernel_ch =
                        kernel_channel_base + in_ch;

                    for (std::size_t kr = 0;
                         kr < kernel_size;
                         ++kr)
                    {
                        const auto input_r =
                            base_r + static_cast<std::ptrdiff_t>(kr);

                        if (input_r < 0 ||
                            input_r >= static_cast<std::ptrdiff_t>(rows_))
                            continue;

                        for (std::size_t kc = 0;
                             kc < kernel_size;
                             ++kc)
                        {
                            const auto input_c =
                                base_c + static_cast<std::ptrdiff_t>(kc);

                            if (input_c < 0 ||
                                input_c >= static_cast<std::ptrdiff_t>(cols_))
                                continue;

                            sum +=
                                (*this)(
                                    in_ch,
                                    static_cast<std::size_t>(input_r),
                                    static_cast<std::size_t>(input_c))
                                *
                                kernels(
                                    kernel_ch,
                                    kr,
                                    kc);
                        }
                    }
                }

                output(out_ch, out_r, out_c) =
                    std::max(0.0, sum);
            }
        }
    }

    return output;
}

} // namespace bpy
