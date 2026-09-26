#include <CNN.hpp>
#include <Conv2D.hpp>
#include <Tensor.hpp>

#include <cstddef>
#include <print>
#include <utility>

using bpy::CNN;
using bpy::Conv2D;
using bpy::Padding;
using bpy::Tensor3D;

Tensor3D make_thermal_image(
    std::size_t rows,
    std::size_t cols,
    std::size_t channels)
{
    Tensor3D image{
        channels,
        rows,
        cols,
        0.1
    };

    // Head.
    for (std::size_t r = 5; r <= 9 && r < rows; ++r)
    {
        for (std::size_t c = 10; c <= 15 && c < cols; ++c)
            image(0, r, c) = 1.0;
    }

    // Torso.
    for (std::size_t r = 10; r <= 23 && r < rows; ++r)
    {
        for (std::size_t c = 8; c <= 17 && c < cols; ++c)
            image(0, r, c) = 0.9;
    }

    // Arms.
    for (std::size_t r = 11; r <= 20 && r < rows; ++r)
    {
        if (6 < cols)
            image(0, r, 6) = 0.8;

        if (7 < cols)
            image(0, r, 7) = 0.8;

        if (18 < cols)
            image(0, r, 18) = 0.8;

        if (19 < cols)
            image(0, r, 19) = 0.8;
    }

    // Legs.
    for (std::size_t r = 24; r < rows; ++r)
    {
        for (std::size_t c = 9; c <= 12 && c < cols; ++c)
            image(0, r, c) = 0.85;

        for (std::size_t c = 14; c <= 17 && c < cols; ++c)
            image(0, r, c) = 0.85;
    }

    return image;
}

int main()
{
    constexpr std::size_t input_rows{32};
    constexpr std::size_t input_cols{24};
    constexpr std::size_t input_channels{1};

    constexpr std::size_t filters{4};
    constexpr std::size_t kernel_size{3};
    constexpr std::size_t convolution_stride{1};

    constexpr std::size_t pool_size{2};
    constexpr std::size_t pool_stride{2};

    constexpr std::size_t hidden_neurons{32};

    // --------------------------------------------------------
    // CNN
    // --------------------------------------------------------

    Conv2D convolution{
        input_channels,
        filters,
        kernel_size,
        convolution_stride,
        Padding::Valid
    };

    const auto convolution_rows =
        convolution.OutputRows(input_rows);

    const auto convolution_cols =
        convolution.OutputCols(input_cols);

    const auto pooled_rows =
        (convolution_rows - pool_size) /
            pool_stride + 1;

    const auto pooled_cols =
        (convolution_cols - pool_size) /
            pool_stride + 1;

    const auto flattened_size =
        filters *
        pooled_rows *
        pooled_cols;

    CNN model{
        std::move(convolution),
        flattened_size,
        hidden_neurons,
        pool_size,
        pool_stride
    };

    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    const Tensor3D thermal_image{
        make_thermal_image(
            input_rows,
            input_cols,
            input_channels
        )
    };

    // --------------------------------------------------------
    // Prediction
    // --------------------------------------------------------

    const double probability =
        model.predict(thermal_image);

    // --------------------------------------------------------
    // Result
    // --------------------------------------------------------

    std::println();

    std::println(
        "Input:       {} x {} x {}",
        thermal_image.Rows(),
        thermal_image.Cols(),
        thermal_image.Channels()
    );

    std::println(
        "Convolution: {} x {} x {}",
        convolution_rows,
        convolution_cols,
        filters
    );

    std::println(
        "Pooling:     {} x {} x {}",
        pooled_rows,
        pooled_cols,
        filters
    );

    std::println(
        "Flattened:   {}",
        flattened_size
    );

    std::println(
        "Dense:       {} -> {} -> 1",
        flattened_size,
        hidden_neurons
    );

    std::println();

    std::println(
        "P(human) = {:.4f}",
        probability
    );

    std::println(
        "P(human) = {:.2f}%",
        probability * 100.0
    );

    std::println(
        "Prediction: {}",
        probability >= 0.5
            ? "HUMAN DETECTED"
            : "NO HUMAN DETECTED"
    );
}
