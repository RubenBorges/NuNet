#include <Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <random>
#include <stdexcept>

#if defined(NUNET_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace bpy {

namespace {

#if defined(NUNET_ENABLE_SYCL)
Tensor3D relu_sycl(const Tensor3D& input)
{
    std::vector<double> source(input.Data(), input.Data() + input.Size());
    Tensor3D result{input.Channels(), input.Rows(), input.Cols()};
    sycl::queue queue{sycl::gpu_selector_v};

    sycl::buffer<double, 1> source_buffer{
        source.data(), sycl::range<1>{source.size()}};
    sycl::buffer<double, 1> result_buffer{
        sycl::range<1>{result.Size()}};

    queue.submit([&](sycl::handler& handler) {
        auto source_values =
            source_buffer.get_access<sycl::access::mode::read>(handler);
        auto result_values =
            result_buffer.get_access<sycl::access::mode::write>(handler);
        handler.parallel_for(sycl::range<1>{result.Size()}, [=](sycl::id<1> id) {
            result_values[id] = sycl::fmax(0.0, source_values[id]);
        });
    });
    queue.wait_and_throw();

    sycl::host_accessor result_values{result_buffer, sycl::read_only};
    for (std::size_t index = 0; index < result.Size(); ++index)
        result.Data()[index] = result_values[index];

    return result;
}

Tensor3D max_pool_sycl(
    const Tensor3D& input,
    std::size_t pool_size,
    std::size_t stride)
{
    const std::size_t output_rows =
        (input.Rows() - pool_size) / stride + 1;
    const std::size_t output_cols =
        (input.Cols() - pool_size) / stride + 1;
    const std::size_t input_rows = input.Rows();
    const std::size_t input_cols = input.Cols();
    Tensor3D result{input.Channels(), output_rows, output_cols};
    std::vector<double> source(input.Data(), input.Data() + input.Size());
    sycl::queue queue{sycl::gpu_selector_v};

    sycl::buffer<double, 1> source_buffer{
        source.data(), sycl::range<1>{source.size()}};
    sycl::buffer<double, 1> result_buffer{
        sycl::range<1>{result.Size()}};

    queue.submit([&](sycl::handler& handler) {
        auto source_values =
            source_buffer.get_access<sycl::access::mode::read>(handler);
        auto result_values =
            result_buffer.get_access<sycl::access::mode::write>(handler);
        handler.parallel_for(sycl::range<1>{result.Size()}, [=](sycl::id<1> id) {
            const std::size_t index = id[0];
            const std::size_t output_plane = output_rows * output_cols;
            const std::size_t channel = index / output_plane;
            const std::size_t row = (index / output_cols) % output_rows;
            const std::size_t col = index % output_cols;
            double maximum = -sycl::INFINITY;

            for (std::size_t pool_row = 0; pool_row < pool_size; ++pool_row)
            {
                for (std::size_t pool_col = 0; pool_col < pool_size; ++pool_col)
                {
                    const std::size_t source_index =
                        (channel * input_rows + row * stride + pool_row)
                            * input_cols
                        + col * stride + pool_col;
                    maximum = sycl::fmax(maximum, source_values[source_index]);
                }
            }

            result_values[index] = maximum;
        });
    });
    queue.wait_and_throw();

    sycl::host_accessor result_values{result_buffer, sycl::read_only};
    for (std::size_t index = 0; index < result.Size(); ++index)
        result.Data()[index] = result_values[index];

    return result;
}

Tensor3D convolve_sycl(
    const Tensor3D& input,
    const Tensor3D& kernels,
    const std::vector<double>& biases,
    std::size_t stride,
    std::size_t padding,
    std::size_t output_rows,
    std::size_t output_cols)
{
    const std::size_t kernel_size = kernels.Rows();
    Tensor3D output{kernels.Channels(), output_rows, output_cols};
    std::vector<double> source(input.Data(), input.Data() + input.Size());
    std::vector<double> weights(kernels.Data(), kernels.Data() + kernels.Size());
    std::vector<double> bias_copy{biases};
    sycl::queue queue{sycl::gpu_selector_v};

    sycl::buffer<double, 1> source_buffer{
        source.data(), sycl::range<1>{source.size()}};
    sycl::buffer<double, 1> kernel_buffer{
        weights.data(), sycl::range<1>{weights.size()}};
    sycl::buffer<double, 1> bias_buffer{
        bias_copy.data(), sycl::range<1>{bias_copy.size()}};
    sycl::buffer<double, 1> output_buffer{
        sycl::range<1>{output.Size()}};

    const std::size_t channels = input.Channels();
    const std::size_t input_rows = input.Rows();
    const std::size_t input_cols = input.Cols();
    const std::size_t output_plane = output_rows * output_cols;

    queue.submit([&](sycl::handler& handler) {
        auto source_values =
            source_buffer.get_access<sycl::access::mode::read>(handler);
        auto kernel_values =
            kernel_buffer.get_access<sycl::access::mode::read>(handler);
        auto bias_values =
            bias_buffer.get_access<sycl::access::mode::read>(handler);
        auto output_values =
            output_buffer.get_access<sycl::access::mode::write>(handler);
        handler.parallel_for(sycl::range<1>{output.Size()}, [=](sycl::id<1> id) {
            const std::size_t index = id[0];
            const std::size_t output_channel = index / output_plane;
            const std::size_t output_row =
                (index / output_cols) % output_rows;
            const std::size_t output_col = index % output_cols;
            const std::ptrdiff_t base_row =
                static_cast<std::ptrdiff_t>(output_row * stride)
                - static_cast<std::ptrdiff_t>(padding);
            const std::ptrdiff_t base_col =
                static_cast<std::ptrdiff_t>(output_col * stride)
                - static_cast<std::ptrdiff_t>(padding);
            double sum = bias_values[output_channel];

            for (std::size_t input_channel = 0;
                 input_channel < channels;
                 ++input_channel)
            {
                const std::size_t kernel_channel =
                    output_channel * channels + input_channel;
                for (std::size_t kernel_row = 0;
                     kernel_row < kernel_size;
                     ++kernel_row)
                {
                    const std::ptrdiff_t input_row =
                        base_row + static_cast<std::ptrdiff_t>(kernel_row);
                    if (input_row < 0 ||
                        input_row >= static_cast<std::ptrdiff_t>(input_rows))
                        continue;

                    for (std::size_t kernel_col = 0;
                         kernel_col < kernel_size;
                         ++kernel_col)
                    {
                        const std::ptrdiff_t input_col =
                            base_col + static_cast<std::ptrdiff_t>(kernel_col);
                        if (input_col < 0 ||
                            input_col >= static_cast<std::ptrdiff_t>(input_cols))
                            continue;

                        const std::size_t source_index =
                            (input_channel * input_rows
                             + static_cast<std::size_t>(input_row))
                                * input_cols
                            + static_cast<std::size_t>(input_col);
                        const std::size_t kernel_index =
                            (kernel_channel * kernel_size + kernel_row)
                                * kernel_size
                            + kernel_col;
                        sum += source_values[source_index]
                            * kernel_values[kernel_index];
                    }
                }
            }

            output_values[index] = sycl::fmax(0.0, sum);
        });
    });
    queue.wait_and_throw();

    sycl::host_accessor output_values{output_buffer, sycl::read_only};
    for (std::size_t index = 0; index < output.Size(); ++index)
        output.Data()[index] = output_values[index];

    return output;
}
#endif

} // namespace

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

Tensor3D Tensor3D::relu(ExecutionPolicy policy) const
{
#if defined(NUNET_ENABLE_SYCL)
    if (policy == ExecutionPolicy::SYCLGPU)
        return relu_sycl(*this);
#else
    if (policy == ExecutionPolicy::SYCLGPU)
        throw std::runtime_error("NuNet was built without SYCL support");
#endif

    Tensor3D result{channels_, rows_, cols_};

    const auto relu_value = [](double value) noexcept {
        return std::max(0.0, value);
    };

    if (policy == ExecutionPolicy::ParallelCPU)
    {
        std::transform(
            std::execution::par_unseq,
            data_.begin(), data_.end(), result.data_.begin(), relu_value);
    }
    else
    {
        detail::simd_relu(data_.data(), result.data_.data(), data_.size());
    }

    return result;
}

Sender<Tensor3D> Tensor3D::relu_async(ExecutionPolicy policy) const
{
    Tensor3D input{*this};
    return Sender<Tensor3D>::submit(
        [input = std::move(input), policy] { return input.relu(policy); });
}

Tensor3D Tensor3D::max_pool(
    std::size_t pool_size,
    std::size_t stride,
    ExecutionPolicy policy) const
{
    if (pool_size == 0 || stride == 0)
        throw std::invalid_argument("Pool size and stride must be greater than zero");

    if (pool_size > rows_ || pool_size > cols_)
        throw std::invalid_argument("Pool window is larger than tensor");

#if defined(NUNET_ENABLE_SYCL)
    if (policy == ExecutionPolicy::SYCLGPU)
        return max_pool_sycl(*this, pool_size, stride);
#else
    if (policy == ExecutionPolicy::SYCLGPU)
        throw std::runtime_error("NuNet was built without SYCL support");
#endif

    const std::size_t output_rows =
        (rows_ - pool_size) / stride + 1;

    const std::size_t output_cols =
        (cols_ - pool_size) / stride + 1;

    Tensor3D result{
        channels_,
        output_rows,
        output_cols
    };

    if (policy == ExecutionPolicy::ParallelCPU)
    {
        std::for_each(
            std::execution::par_unseq,
            result.data_.begin(), result.data_.end(),
            [&](double& value) {
                const std::size_t index =
                    static_cast<std::size_t>(&value - result.data_.data());
                const std::size_t plane = output_rows * output_cols;
                const std::size_t channel = index / plane;
                const std::size_t row = (index / output_cols) % output_rows;
                const std::size_t col = index % output_cols;
                double maximum = -std::numeric_limits<double>::infinity();

                for (std::size_t pool_row = 0; pool_row < pool_size; ++pool_row)
                {
                    for (std::size_t pool_col = 0; pool_col < pool_size; ++pool_col)
                    {
                        maximum = std::max(
                            maximum,
                            (*this)(
                                channel,
                                row * stride + pool_row,
                                col * stride + pool_col));
                    }
                }

                value = maximum;
            });
        return result;
    }

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

Sender<Tensor3D> Tensor3D::max_pool_async(
    std::size_t pool_size,
    std::size_t stride,
    ExecutionPolicy policy) const
{
    Tensor3D input{*this};
    return Sender<Tensor3D>::submit(
        [input = std::move(input), pool_size, stride, policy] {
            return input.max_pool(pool_size, stride, policy);
        });
}

std::vector<double> Tensor3D::flatten(ExecutionPolicy policy) const
{
    std::vector<double> result(data_.size());

    if (policy == ExecutionPolicy::ParallelCPU)
        std::copy(std::execution::par_unseq, data_.begin(), data_.end(), result.begin());
    else
        std::copy(data_.begin(), data_.end(), result.begin());

    return result;
}

Sender<std::vector<double>> Tensor3D::flatten_async(
    ExecutionPolicy policy) const
{
    Tensor3D input{*this};
    return Sender<std::vector<double>>::submit(
        [input = std::move(input), policy] { return input.flatten(policy); });
}

Tensor3D Tensor3D::convolve(
    const Tensor3D& kernels,
    const std::vector<double>& biases,
    std::size_t stride,
    std::size_t padding,
    ExecutionPolicy policy) const
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

#if defined(NUNET_ENABLE_SYCL)
    if (policy == ExecutionPolicy::SYCLGPU)
    {
        return convolve_sycl(
            *this, kernels, biases, stride, padding,
            output_rows, output_cols);
    }
#else
    if (policy == ExecutionPolicy::SYCLGPU)
        throw std::runtime_error("NuNet was built without SYCL support");
#endif

    if (policy == ExecutionPolicy::ParallelCPU)
    {
        const std::size_t output_plane = output_rows * output_cols;
        std::for_each(
            std::execution::par_unseq,
            output.data_.begin(), output.data_.end(),
            [&](double& value) {
                const std::size_t index =
                    static_cast<std::size_t>(&value - output.data_.data());
                const std::size_t output_channel = index / output_plane;
                const std::size_t output_row = (index / output_cols) % output_rows;
                const std::size_t output_col = index % output_cols;
                const std::ptrdiff_t base_row =
                    static_cast<std::ptrdiff_t>(output_row * stride)
                    - static_cast<std::ptrdiff_t>(padding);
                const std::ptrdiff_t base_col =
                    static_cast<std::ptrdiff_t>(output_col * stride)
                    - static_cast<std::ptrdiff_t>(padding);
                double sum = biases[output_channel];

                for (std::size_t input_channel = 0;
                     input_channel < channels_;
                     ++input_channel)
                {
                    const std::size_t kernel_channel =
                        output_channel * channels_ + input_channel;
                    for (std::size_t kernel_row = 0;
                         kernel_row < kernel_size;
                         ++kernel_row)
                    {
                        const std::ptrdiff_t input_row =
                            base_row + static_cast<std::ptrdiff_t>(kernel_row);
                        if (input_row < 0 ||
                            input_row >= static_cast<std::ptrdiff_t>(rows_))
                            continue;

                        for (std::size_t kernel_col = 0;
                             kernel_col < kernel_size;
                             ++kernel_col)
                        {
                            const std::ptrdiff_t input_col =
                                base_col + static_cast<std::ptrdiff_t>(kernel_col);
                            if (input_col < 0 ||
                                input_col >= static_cast<std::ptrdiff_t>(cols_))
                                continue;

                            sum += (*this)(
                                       input_channel,
                                       static_cast<std::size_t>(input_row),
                                       static_cast<std::size_t>(input_col))
                                * kernels(
                                    kernel_channel,
                                    kernel_row,
                                    kernel_col);
                        }
                    }
                }

                value = std::max(0.0, sum);
            });
        return output;
    }

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

Sender<Tensor3D> Tensor3D::convolve_async(
    const Tensor3D& kernels,
    const std::vector<double>& biases,
    std::size_t stride,
    std::size_t padding,
    ExecutionPolicy policy) const
{
    Tensor3D input{*this};
    Tensor3D kernel_copy{kernels};
    std::vector<double> bias_copy{biases};
    return Sender<Tensor3D>::submit(
        [input = std::move(input),
         kernels = std::move(kernel_copy),
         biases = std::move(bias_copy),
         stride, padding, policy] {
            return input.convolve(kernels, biases, stride, padding, policy);
        });
}

} // namespace bpy
