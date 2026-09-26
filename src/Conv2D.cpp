#include <Conv2D.hpp>

#include <cmath>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>

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
          input_channels * output_channels,
          kernel_size,
          kernel_size
      },
      biases_(output_channels, 0.0)
{
    if (input_channels == 0 ||
        output_channels == 0 ||
        kernel_size == 0 ||
        stride == 0)
    {
        throw std::invalid_argument(
            "Conv2D dimensions must be greater than zero");
    }

    randomize();
}

void Conv2D::randomize()
{
    /*
        He initialization.

        fan_in = input channels * kernel area

        std::sqrt(2 / fan_in)
    */

    const double fan_in =
        static_cast<double>(
            input_channels_ *
            kernel_size_ *
            kernel_size_);

    const double limit =
        std::sqrt(6.0 / fan_in);

    weights_.randomize(-limit, limit);

    std::fill(
        biases_.begin(),
        biases_.end(),
        0.0);
}

Tensor3D Conv2D::forward(
    const Tensor3D& input,
    ExecutionPolicy policy) const
{
    if (input.Channels() != input_channels_)
    {
        throw std::invalid_argument(
            "Conv2D input channel count does not match layer");
    }

    const std::size_t padding =
        padding_ == Padding::Same
            ? kernel_size_ / 2
            : 0;

    return input.convolve(
        weights_,
        biases_,
        stride_,
        padding,
        policy);
}

Sender<Tensor3D> Conv2D::forward_async(
    const Tensor3D& input,
    ExecutionPolicy policy) const
{
    if (input.Channels() != input_channels_)
    {
        throw std::invalid_argument(
            "Conv2D input channel count does not match layer");
    }

    const std::size_t padding =
        padding_ == Padding::Same
            ? kernel_size_ / 2
            : 0;

    return input.convolve_async(
        weights_, biases_, stride_, padding, policy);
}

void Conv2D::save(std::ostream& stream) const
{
    stream.write(
        reinterpret_cast<const char*>(&input_channels_),
        sizeof(input_channels_));

    stream.write(
        reinterpret_cast<const char*>(&output_channels_),
        sizeof(output_channels_));

    stream.write(
        reinterpret_cast<const char*>(&kernel_size_),
        sizeof(kernel_size_));

    stream.write(
        reinterpret_cast<const char*>(&stride_),
        sizeof(stride_));

    const auto padding =
        static_cast<std::uint8_t>(padding_);

    stream.write(
        reinterpret_cast<const char*>(&padding),
        sizeof(padding));

    const auto weight_count =
        static_cast<std::uint64_t>(weights_.Size());

    stream.write(
        reinterpret_cast<const char*>(&weight_count),
        sizeof(weight_count));

    stream.write(
        reinterpret_cast<const char*>(weights_.Data()),
        static_cast<std::streamsize>(
            weights_.Size() * sizeof(double)));

    const auto bias_count =
        static_cast<std::uint64_t>(biases_.size());

    stream.write(
        reinterpret_cast<const char*>(&bias_count),
        sizeof(bias_count));

    stream.write(
        reinterpret_cast<const char*>(biases_.data()),
        static_cast<std::streamsize>(
            biases_.size() * sizeof(double)));
}

void Conv2D::load(std::istream& stream)
{
    std::size_t input_channels{};
    std::size_t output_channels{};
    std::size_t kernel_size{};
    std::size_t stride{};
    std::uint8_t padding{};

    stream.read(
        reinterpret_cast<char*>(&input_channels),
        sizeof(input_channels));

    stream.read(
        reinterpret_cast<char*>(&output_channels),
        sizeof(output_channels));

    stream.read(
        reinterpret_cast<char*>(&kernel_size),
        sizeof(kernel_size));

    stream.read(
        reinterpret_cast<char*>(&stride),
        sizeof(stride));

    stream.read(
        reinterpret_cast<char*>(&padding),
        sizeof(padding));

    if (!stream)
        throw std::runtime_error("Corrupt Conv2D model data");

    if (input_channels != input_channels_ ||
        output_channels != output_channels_ ||
        kernel_size != kernel_size_ ||
        stride != stride_ ||
        padding != static_cast<std::uint8_t>(padding_))
    {
        throw std::runtime_error(
            "Conv2D architecture does not match saved model");
    }

    std::uint64_t weight_count{};

    stream.read(
        reinterpret_cast<char*>(&weight_count),
        sizeof(weight_count));

    if (weight_count != weights_.Size())
        throw std::runtime_error("Invalid Conv2D weight count");

    stream.read(
        reinterpret_cast<char*>(weights_.Data()),
        static_cast<std::streamsize>(
            weights_.Size() * sizeof(double)));

    std::uint64_t bias_count{};

    stream.read(
        reinterpret_cast<char*>(&bias_count),
        sizeof(bias_count));

    if (bias_count != biases_.size())
        throw std::runtime_error("Invalid Conv2D bias count");

    stream.read(
        reinterpret_cast<char*>(biases_.data()),
        static_cast<std::streamsize>(
            biases_.size() * sizeof(double)));

    if (!stream)
        throw std::runtime_error("Corrupt Conv2D weights");
}

} // namespace bpy
