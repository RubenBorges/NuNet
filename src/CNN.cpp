#include <CNN.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bpy {

double CNN::relu(double x) noexcept
{
    return std::max(0.0, x);
}

double CNN::sigmoid(double x) noexcept
{
    // Numerically stable sigmoid.
    if (x >= 0.0)
    {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }

    const double z = std::exp(x);
    return z / (1.0 + z);
}

CNN::CNN(
    Conv2D convolution,
    std::size_t flattened_size,
    std::size_t hidden_neurons,
    std::size_t pool_size,
    std::size_t pool_stride)
    : convolution_{std::move(convolution)},
      pool_size_{pool_size},
      pool_stride_{pool_stride},
      hidden_{
          flattened_size,
          hidden_neurons,
          relu
      },
      output_{
          hidden_neurons,
          1,
          sigmoid
      }
{
    if (flattened_size == 0)
        throw std::invalid_argument(
            "CNN flattened size must be greater than zero");

    if (hidden_neurons == 0)
        throw std::invalid_argument(
            "CNN hidden layer must contain neurons");

    if (pool_size == 0 ||
        pool_stride == 0)
        throw std::invalid_argument(
            "CNN pooling parameters must be greater than zero");
}

Tensor3D CNN::max_pool(
    const Tensor3D& input) const
{
    if (input.Rows() < pool_size_ ||
        input.Cols() < pool_size_)
        throw std::invalid_argument(
            "Pooling window is larger than feature map");

    const std::size_t output_rows =
        (input.Rows() - pool_size_) /
            pool_stride_ + 1;

    const std::size_t output_cols =
        (input.Cols() - pool_size_) /
            pool_stride_ + 1;

    Tensor3D output{
        input.Channels(),
        output_rows,
        output_cols,
        0.0
    };

    for (std::size_t channel = 0;
         channel < input.Channels();
         ++channel)
    {
        for (std::size_t r = 0;
             r < output_rows;
             ++r)
        {
            for (std::size_t c = 0;
                 c < output_cols;
                 ++c)
            {
                double maximum =
                    -std::numeric_limits<double>::infinity();

                const std::size_t input_r =
                    r * pool_stride_;

                const std::size_t input_c =
                    c * pool_stride_;

                for (std::size_t pr = 0;
                     pr < pool_size_;
                     ++pr)
                {
                    for (std::size_t pc = 0;
                         pc < pool_size_;
                         ++pc)
                    {
                        maximum = std::max(
                            maximum,
                            input(
                                channel,
                                input_r + pr,
                                input_c + pc
                            )
                        );
                    }
                }

                output(
                    channel,
                    r,
                    c
                ) = maximum;
            }
        }
    }

    return output;
}

Matrix CNN::flatten(
    const Tensor3D& input) const
{
    Matrix output(
        1,
        input.Channels() *
        input.Rows() *
        input.Cols(),
        0.0
    );

    std::size_t index = 0;

    for (std::size_t channel = 0;
         channel < input.Channels();
         ++channel)
    {
        for (std::size_t r = 0;
             r < input.Rows();
             ++r)
        {
            for (std::size_t c = 0;
                 c < input.Cols();
                 ++c)
            {
                output(0, index++) =
                    input(channel, r, c);
            }
        }
    }

    return output;
}

double CNN::predict(
    const Tensor3D& input) const
{
    if (input.Channels() !=
        convolution_.InputChannels())
    {
        throw std::invalid_argument(
            "CNN input channel count does not match convolution");
    }

    // Input
    //
    // 32 x 24 x 1
    //
    //      ↓
    //
    // Conv
    //
    // 30 x 22 x 4
    //
    const Tensor3D features =
        convolution_.forward(input);

    //      ↓
    //
    // MaxPool
    //
    // 15 x 11 x 4
    //
    const Tensor3D pooled =
        max_pool(features);

    //      ↓
    //
    // Flatten
    //
    // 660
    //
    const Matrix flattened =
        flatten(pooled);

    if (flattened.Cols() !=
        hidden_.InputSize())
    {
        throw std::runtime_error(
            "CNN architecture mismatch: "
            "flattened feature count does not match dense layer"
        );
    }

    //      ↓
    //
    // Dense
    //
    // 660 -> 32
    //
    const Matrix hidden =
        hidden_.forward(flattened);

    //      ↓
    //
    // Dense
    //
    // 32 -> 1
    //
    const Matrix prediction =
        output_.forward(hidden);

    return prediction(0, 0);
}

} // namespace bpy
