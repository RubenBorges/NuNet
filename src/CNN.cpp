#include <CNN.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace bpy {

namespace {

constexpr std::uint64_t MODEL_MAGIC =
    0x4E554E45544D4F44ULL; // "NUNETMOD"

constexpr std::uint32_t MODEL_VERSION = 1;

}

double CNN::relu(double value) noexcept
{
    return std::max(0.0, value);
}

double CNN::sigmoid(double value) noexcept
{
    /*
        Numerically stable sigmoid.
    */

    if (value >= 0.0)
    {
        const double z =
            std::exp(-value);

        return 1.0 / (1.0 + z);
    }

    const double z =
        std::exp(value);

    return z / (1.0 + z);
}

CNN::CNN(
    Config config,
    std::size_t input_rows,
    std::size_t input_cols)
    : config_{config},
      input_rows_{input_rows},
      input_cols_{input_cols},
      flattened_size_{},
      convolution_{
          config_.input_channels,
          config_.filters,
          config_.kernel_size,
          config_.convolution_stride,
          config_.padding
      },
      hidden_{
          1,
          config_.hidden_size,
          relu
      },
      output_{
          config_.hidden_size,
          1,
          sigmoid
      }
{
    if (input_rows == 0 ||
        input_cols == 0)
    {
        throw std::invalid_argument(
            "CNN input dimensions must be greater than zero");
    }

    if (config_.input_channels == 0 ||
        config_.filters == 0 ||
        config_.kernel_size == 0 ||
        config_.convolution_stride == 0 ||
        config_.pool_size == 0 ||
        config_.pool_stride == 0 ||
        config_.hidden_size == 0)
    {
        throw std::invalid_argument(
            "CNN configuration contains zero dimensions");
    }

    flattened_size_ =
        calculate_flattened_size();

    /*
        The hidden layer depends on the flattened convolution
        output, so construct it with the actual size.
    */

    hidden_ =
        DenseLayer{
            flattened_size_,
            config_.hidden_size,
            relu
        };

    output_ =
        DenseLayer{
            config_.hidden_size,
            1,
            sigmoid
        };
}

std::size_t CNN::calculate_flattened_size() const
{
    std::size_t padding = 0;

    if (config_.padding == Padding::Same)
        padding = config_.kernel_size / 2;

    if (input_rows_ + 2 * padding <
            config_.kernel_size ||
        input_cols_ + 2 * padding <
            config_.kernel_size)
    {
        throw std::invalid_argument(
            "CNN kernel is larger than input");
    }

    const std::size_t conv_rows =
        (input_rows_ +
         2 * padding -
         config_.kernel_size)
        / config_.convolution_stride + 1;

    const std::size_t conv_cols =
        (input_cols_ +
         2 * padding -
         config_.kernel_size)
        / config_.convolution_stride + 1;

    if (conv_rows < config_.pool_size ||
        conv_cols < config_.pool_size)
    {
        throw std::invalid_argument(
            "Pooling window is larger than convolution output");
    }

    const std::size_t pooled_rows =
        (conv_rows - config_.pool_size)
        / config_.pool_stride + 1;

    const std::size_t pooled_cols =
        (conv_cols - config_.pool_size)
        / config_.pool_stride + 1;

    return pooled_rows *
           pooled_cols *
           config_.filters;
}

void CNN::validate_input(
    const Tensor3D& input) const
{
    if (input.Channels() != config_.input_channels ||
        input.Rows() != input_rows_ ||
        input.Cols() != input_cols_)
    {
        throw std::invalid_argument(
            "Input tensor dimensions do not match CNN configuration");
    }
}

Tensor3D CNN::convolution_features(
    const Tensor3D& input) const
{
    validate_input(input);

    return convolution_.forward(input);
}

Tensor3D CNN::pooled_features(
    const Tensor3D& input) const
{
    const Tensor3D convolution =
        convolution_features(input);

    const Tensor3D activated = convolution.relu();

    return activated.max_pool(
        config_.pool_size,
        config_.pool_stride);
}

double CNN::predict(
    const Tensor3D& input) const
{
    const Tensor3D convolution =
        convolution_.forward(input);

    const Tensor3D pooled =
        convolution.max_pool(
            config_.pool_size,
            config_.pool_stride);

    const auto flattened =
        pooled.flatten();

    Matrix input_matrix{
        1,
        flattened.size()
    };

    std::copy(
        flattened.begin(),
        flattened.end(),
        input_matrix.Data());

    const Matrix hidden =
        hidden_.forward(input_matrix);

    const Matrix output =
        output_.forward(hidden);

    return output(0, 0);
}

Sender<double> CNN::predict_async(
    const Tensor3D& input,
    ExecutionPolicy policy) const
{
    validate_input(input);

    auto convolution =
        Sender<Tensor3D>::just(input).let_value(
            [layer = convolution_, policy](Tensor3D image) {
                return layer.forward_async(image, policy);
            });

    auto pooled =
        std::move(convolution).let_value(
            [pool_size = config_.pool_size,
             pool_stride = config_.pool_stride,
             policy](Tensor3D features) {
                return features.max_pool_async(
                    pool_size, pool_stride, policy);
            });

    auto flattened =
        std::move(pooled).let_value(
            [policy](Tensor3D features) {
                return features.flatten_async(policy);
            });

    auto dense_input =
        std::move(flattened).then(
            [](std::vector<double> values) {
                Matrix matrix{1, values.size()};
                std::copy(values.begin(), values.end(), matrix.Data());
                return matrix;
            });

    auto hidden =
        std::move(dense_input).let_value(
            [layer = hidden_, policy](Matrix values) {
                return layer.forward_async(values, policy);
            });

    auto output =
        std::move(hidden).let_value(
            [layer = output_, policy](Matrix values) {
                return layer.forward_async(values, policy);
            });

    return std::move(output).then(
        [](Matrix probabilities) { return probabilities(0, 0); });
}

void CNN::randomize()
{
    convolution_.randomize();
    hidden_.randomize();
    output_.randomize();
}

void CNN::save(
    const std::filesystem::path& filename) const
{
    std::ofstream stream{
        filename,
        std::ios::binary
    };

    if (!stream)
        throw std::runtime_error(
            "Could not open model for writing: " +
            filename.string());

    stream.write(
        reinterpret_cast<const char*>(&MODEL_MAGIC),
        sizeof(MODEL_MAGIC));

    stream.write(
        reinterpret_cast<const char*>(&MODEL_VERSION),
        sizeof(MODEL_VERSION));

    const auto input_rows =
        static_cast<std::uint64_t>(input_rows_);

    const auto input_cols =
        static_cast<std::uint64_t>(input_cols_);

    stream.write(
        reinterpret_cast<const char*>(&input_rows),
        sizeof(input_rows));

    stream.write(
        reinterpret_cast<const char*>(&input_cols),
        sizeof(input_cols));

    const auto input_channels =
        static_cast<std::uint64_t>(
            config_.input_channels);

    const auto filters =
        static_cast<std::uint64_t>(
            config_.filters);

    const auto kernel_size =
        static_cast<std::uint64_t>(
            config_.kernel_size);

    const auto convolution_stride =
        static_cast<std::uint64_t>(
            config_.convolution_stride);

    const auto pool_size =
        static_cast<std::uint64_t>(
            config_.pool_size);

    const auto pool_stride =
        static_cast<std::uint64_t>(
            config_.pool_stride);

    const auto hidden_size =
        static_cast<std::uint64_t>(
            config_.hidden_size);

    const auto padding =
        static_cast<std::uint8_t>(
            config_.padding);

    stream.write(
        reinterpret_cast<const char*>(&input_channels),
        sizeof(input_channels));

    stream.write(
        reinterpret_cast<const char*>(&filters),
        sizeof(filters));

    stream.write(
        reinterpret_cast<const char*>(&kernel_size),
        sizeof(kernel_size));

    stream.write(
        reinterpret_cast<const char*>(&convolution_stride),
        sizeof(convolution_stride));

    stream.write(
        reinterpret_cast<const char*>(&pool_size),
        sizeof(pool_size));

    stream.write(
        reinterpret_cast<const char*>(&pool_stride),
        sizeof(pool_stride));

    stream.write(
        reinterpret_cast<const char*>(&hidden_size),
        sizeof(hidden_size));

    stream.write(
        reinterpret_cast<const char*>(&padding),
        sizeof(padding));

    convolution_.save(stream);
    hidden_.save(stream);
    output_.save(stream);

    if (!stream)
        throw std::runtime_error(
            "Failed while writing model: " +
            filename.string());
}

CNN CNN::load_model(
    const std::filesystem::path& filename)
{
    std::ifstream stream{
        filename,
        std::ios::binary
    };

    if (!stream)
        throw std::runtime_error(
            "Could not open model: " +
            filename.string());

    std::uint64_t magic{};
    std::uint32_t version{};

    stream.read(
        reinterpret_cast<char*>(&magic),
        sizeof(magic));

    stream.read(
        reinterpret_cast<char*>(&version),
        sizeof(version));

    if (magic != MODEL_MAGIC)
        throw std::runtime_error(
            "Invalid NuNet model file");

    if (version != MODEL_VERSION)
        throw std::runtime_error(
            "Unsupported NuNet model version");

    std::uint64_t input_rows{};
    std::uint64_t input_cols{};

    stream.read(
        reinterpret_cast<char*>(&input_rows),
        sizeof(input_rows));

    stream.read(
        reinterpret_cast<char*>(&input_cols),
        sizeof(input_cols));

    Config config{};

    std::uint64_t value{};

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.input_channels = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.filters = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.kernel_size = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.convolution_stride = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.pool_size = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.pool_stride = value;

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    config.hidden_size = value;

    std::uint8_t padding{};

    stream.read(
        reinterpret_cast<char*>(&padding),
        sizeof(padding));

    config.padding =
        static_cast<Padding>(padding);

    if (!stream)
        throw std::runtime_error(
            "Corrupt NuNet model header");

    CNN model{
        config,
        input_rows,
        input_cols
    };

    model.convolution_.load(stream);
    model.hidden_.load(stream);
    model.output_.load(stream);

    return model;
}

void CNN::load(
    const std::filesystem::path& filename)
{
    CNN loaded =
        load_model(filename);

    if (loaded.input_rows_ != input_rows_ ||
        loaded.input_cols_ != input_cols_ ||
        loaded.config_.input_channels !=
            config_.input_channels ||
        loaded.config_.filters !=
            config_.filters ||
        loaded.config_.kernel_size !=
            config_.kernel_size ||
        loaded.config_.convolution_stride !=
            config_.convolution_stride ||
        loaded.config_.padding !=
            config_.padding ||
        loaded.config_.pool_size !=
            config_.pool_size ||
        loaded.config_.pool_stride !=
            config_.pool_stride ||
        loaded.config_.hidden_size !=
            config_.hidden_size)
    {
        throw std::runtime_error(
            "Saved model architecture does not match current CNN");
    }

    convolution_ =
        std::move(loaded.convolution_);

    hidden_ =
        std::move(loaded.hidden_);

    output_ =
        std::move(loaded.output_);
}

} // namespace bpy
