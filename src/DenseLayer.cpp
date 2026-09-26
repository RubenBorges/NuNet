#include <DenseLayer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace bpy {

DenseLayer::DenseLayer(
    std::size_t input_size,
    std::size_t output_size,
    Activation activation)
    : weights_{input_size, output_size},
      biases_{1, output_size},
      activation_{std::move(activation)}
{
    if (input_size == 0 || output_size == 0)
        throw std::invalid_argument(
            "Dense layer dimensions must be greater than zero");

    if (!activation_)
        throw std::invalid_argument(
            "Dense layer requires an activation function");

    randomize();
}

void DenseLayer::randomize()
{
    const double fan_in =
        static_cast<double>(weights_.Rows());

    const double limit =
        std::sqrt(6.0 / fan_in);

    weights_.randomize(-limit, limit);

    for (std::size_t c = 0; c < biases_.Cols(); ++c)
        biases_(0, c) = 0.0;
}

Matrix DenseLayer::forward(const Matrix& input) const
{
    if (input.Cols() != weights_.Rows())
    {
        throw std::invalid_argument(
            "Dense input size does not match layer");
    }

    Matrix output =
        Matrix::dense(
            input,
            weights_,
            biases_);

    output.map_inplace(activation_);

    return output;
}

void DenseLayer::save(std::ostream& stream) const
{
    const auto rows =
        static_cast<std::uint64_t>(weights_.Rows());

    const auto cols =
        static_cast<std::uint64_t>(weights_.Cols());

    stream.write(
        reinterpret_cast<const char*>(&rows),
        sizeof(rows));

    stream.write(
        reinterpret_cast<const char*>(&cols),
        sizeof(cols));

    for (std::size_t r = 0; r < weights_.Rows(); ++r)
    {
        stream.write(
            reinterpret_cast<const char*>(
                &weights_(r, 0)),
            static_cast<std::streamsize>(
                weights_.Cols() * sizeof(double)));
    }

    const auto bias_count =
        static_cast<std::uint64_t>(biases_.Cols());

    stream.write(
        reinterpret_cast<const char*>(&bias_count),
        sizeof(bias_count));

    stream.write(
        reinterpret_cast<const char*>(biases_.Data()),
        static_cast<std::streamsize>(
            biases_.Size() * sizeof(double)));
}

void DenseLayer::load(std::istream& stream)
{
    std::uint64_t rows{};
    std::uint64_t cols{};

    stream.read(
        reinterpret_cast<char*>(&rows),
        sizeof(rows));

    stream.read(
        reinterpret_cast<char*>(&cols),
        sizeof(cols));

    if (rows != weights_.Rows() ||
        cols != weights_.Cols())
    {
        throw std::runtime_error(
            "Dense layer architecture does not match saved model");
    }

    for (std::size_t r = 0; r < weights_.Rows(); ++r)
    {
        stream.read(
            reinterpret_cast<char*>(&weights_(r, 0)),
            static_cast<std::streamsize>(
                weights_.Cols() * sizeof(double)));
    }

    std::uint64_t bias_count{};

    stream.read(
        reinterpret_cast<char*>(&bias_count),
        sizeof(bias_count));

    if (bias_count != biases_.Cols())
        throw std::runtime_error(
            "Invalid Dense bias count");

    stream.read(
        reinterpret_cast<char*>(biases_.Data()),
        static_cast<std::streamsize>(
            biases_.Size() * sizeof(double)));

    if (!stream)
        throw std::runtime_error(
            "Corrupt Dense layer data");
}

} // namespace bpy
