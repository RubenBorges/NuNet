#include <CNN.hpp>
#include <Tensor.hpp>

#include <cstddef>
#include <filesystem>
#include <print>

using bpy::CNN;
using bpy::Padding;
using bpy::Tensor3D;

Tensor3D make_thermal_image(std::size_t rows, std::size_t cols, std::size_t channels) {
  Tensor3D image{channels, rows, cols, 0.1};

  // Head.
  for (std::size_t r = 5; r <= 9 && r < rows; ++r) {
    for (std::size_t c = 10; c <= 15 && c < cols; ++c) image(0, r, c) = 1.0;
  }

  // Torso.
  for (std::size_t r = 10; r <= 23 && r < rows; ++r) {
    for (std::size_t c = 8; c <= 17 && c < cols; ++c) image(0, r, c) = 0.9;
  }

  // Arms.
  for (std::size_t r = 11; r <= 20 && r < rows; ++r) {
    if (6 < cols) image(0, r, 6) = 0.8;
    if (7 < cols) image(0, r, 7) = 0.8;
    if (18 < cols) image(0, r, 18) = 0.8;
    if (19 < cols) image(0, r, 19) = 0.8;
  }

  // Legs.
  for (std::size_t r = 24; r < rows; ++r) {
    for (std::size_t c = 9; c <= 12 && c < cols; ++c) image(0, r, c) = 0.85;
    for (std::size_t c = 14; c <= 17 && c < cols; ++c) image(0, r, c) = 0.85;
  }

  return image;
}

int main() {
  constexpr std::size_t input_rows{32};
  constexpr std::size_t input_cols{24};
  constexpr std::size_t input_channels{1};

  constexpr std::size_t filters{8};
  constexpr std::size_t kernel_size{3};
  constexpr std::size_t hidden_size{32};

  constexpr std::size_t pool_size{2};
  constexpr std::size_t pool_stride{2};

  constexpr auto model_file{"human_detector.nn"};

  // --------------------------------------------------------
  // NETWORK DEFINITION
  // --------------------------------------------------------

  const CNN::Config config{.input_channels = input_channels,

                           .filters = filters,
                           .kernel_size = kernel_size,
                           .convolution_stride = 1,
                           .padding = Padding::Valid,

                           .pool_size = pool_size,
                           .pool_stride = pool_stride,

                           .hidden_size = hidden_size};

  // --------------------------------------------------------
  // MODEL
  // --------------------------------------------------------

  CNN model{config, input_rows, input_cols};

  // --------------------------------------------------------
  // INPUT
  // --------------------------------------------------------

  const Tensor3D thermal_image{make_thermal_image(input_rows, input_cols, input_channels)};

  // --------------------------------------------------------
  // NETWORK INFORMATION
  // --------------------------------------------------------

  const auto convolution_features{ model.convolution_features(thermal_image)};

  const auto pooled_features {model.pooled_features(thermal_image)};

  std::println("Input:        {} x {} x {}", thermal_image.Rows(), thermal_image.Cols(), thermal_image.Channels());

  std::println("Convolution:  {} x {} x {}", convolution_features.Rows(), convolution_features.Cols(), convolution_features.Channels());

  std::println("Pooling:      {} x {} x {}", pooled_features.Rows(), pooled_features.Cols(), pooled_features.Channels());

  std::println("Flattened:    {}", model.FlattenedSize());

  std::println("Hidden:       {}", hidden_size);

  // --------------------------------------------------------
  // PREDICTION
  // --------------------------------------------------------

  const double probability = model.predict(thermal_image);

  std::println();
  std::println("P(human) = {:.4f}", probability);

  std::println("P(human) = {:.2f}%", probability * 100.0);

  std::println("Prediction: {}",
               probability >= 0.5 ? "HUMAN DETECTED" : "NO HUMAN DETECTED");

  // --------------------------------------------------------
  // SAVE MODEL
  // --------------------------------------------------------

  model.save(model_file);

  std::println("Model saved: {}", model_file);

  // --------------------------------------------------------
  // LOAD MODEL
  // --------------------------------------------------------

  CNN loaded { CNN::load_model(model_file)};

  const double loaded_probability = loaded.predict(thermal_image);

  std::println();
  std::println("Loaded model P(human) = {:.4f}", loaded_probability);

  return 0;
}
