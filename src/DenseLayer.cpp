#include <DenseLayer.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bpy {

DenseLayer::DenseLayer(
    std::size_t input_size,
    std::size_t output_size,
    std::function<double(double)> activation)
    : weights_{
          input_size,
          output_size,
          0.0
      },
      biases_{
          1,
          output_size,
          0.0
      },
      activation_{
          std::move(activation)
      }
{
    if (input_size == 0 || output_size == 0)
        throw std::invalid_argument(
            "DenseLayer dimensions must be greater than zero");

    initialize_he(input_size);
}

Matrix DenseLayer::forward(
    const Matrix& inputs) const
{
    if (inputs.Cols() != weights_.Rows())
        throw std::invalid_argument(
            "DenseLayer input dimensions do not match weights");

    Matrix output =
        Matrix::dense(
            inputs,
            weights_,
            biases_
        );

    if (activation_)
        output.map_inplace(activation_);

    return output;
}

void DenseLayer::randomize(
    double min,
    double max)
{
    weights_.randomize(min, max);
    biases_.randomize(
        min * 0.1,
        max * 0.1
    );
}

void DenseLayer::initialize_he(
    std::size_t input_size)
{
    if (input_size == 0)
        throw std::invalid_argument(
            "He initialization requires input_size > 0");

    const double limit =
        std::sqrt(
            6.0 /
            static_cast<double>(input_size)
        );

    weights_.randomize(
        -limit,
        limit
    );

    biases_.map_inplace(
        [](double) noexcept {
            return 0.0;
        }
    );
}

void DenseLayer::save_to_csv(
    const std::string& weights_filename,
    const std::string& biases_filename) const
{
    std::ofstream weights_file(weights_filename);

    if (!weights_file)
        throw std::runtime_error(
            "Could not open weights file: " +
            weights_filename
        );

    for (std::size_t r = 0;
         r < weights_.Rows();
         ++r)
    {
        for (std::size_t c = 0;
             c < weights_.Cols();
             ++c)
        {
            if (c != 0)
                weights_file << ',';

            weights_file << weights_(r, c);
        }

        weights_file << '\n';
    }

    std::ofstream biases_file(biases_filename);

    if (!biases_file)
        throw std::runtime_error(
            "Could not open biases file: " +
            biases_filename
        );

    for (std::size_t c = 0;
         c < biases_.Cols();
         ++c)
    {
        if (c != 0)
            biases_file << ',';

        biases_file << biases_(0, c);
    }

    biases_file << '\n';
}

void DenseLayer::load_from_csv(
    const std::string& weights_filename,
    const std::string& biases_filename)
{
    std::ifstream weights_file(weights_filename);

    if (!weights_file)
        throw std::runtime_error(
            "Could not open weights file: " +
            weights_filename
        );

    std::string line;

    for (std::size_t r = 0;
         r < weights_.Rows();
         ++r)
    {
        if (!std::getline(weights_file, line))
            throw std::runtime_error(
                "Weights file contains too few rows"
            );

        std::stringstream stream(line);
        std::string value;

        for (std::size_t c = 0;
             c < weights_.Cols();
             ++c)
        {
            if (!std::getline(stream, value, ','))
                throw std::runtime_error(
                    "Weights file contains too few columns"
                );

            weights_(r, c) =
                std::stod(value);
        }
    }

    std::ifstream biases_file(biases_filename);

    if (!biases_file)
        throw std::runtime_error(
            "Could not open biases file: " +
            biases_filename
        );

    if (!std::getline(biases_file, line))
        throw std::runtime_error(
            "Bias file is empty"
        );

    std::stringstream stream(line);
    std::string value;

    for (std::size_t c = 0;
         c < biases_.Cols();
         ++c)
    {
        if (!std::getline(stream, value, ','))
            throw std::runtime_error(
                "Bias file contains too few values"
            );

        biases_(0, c) =
            std::stod(value);
    }
}

} // namespace bpy
