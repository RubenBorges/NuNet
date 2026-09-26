#include <Matrix.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace bpy
{
    Matrix::Matrix(
        std::size_t rows,
        std::size_t cols,
        double init_value
    )
        : rows_{rows},
          cols_{cols},
          data_(rows * cols, init_value)
    {
    }

    double& Matrix::operator()(
        std::size_t row,
        std::size_t col
    ) noexcept
    {
        return data_[row * cols_ + col];
    }

    const double& Matrix::operator()(
        std::size_t row,
        std::size_t col
    ) const noexcept
    {
        return data_[row * cols_ + col];
    }

    double Matrix::get(
        std::size_t row,
        std::size_t col
    ) const noexcept
    {
        return data_[row * cols_ + col];
    }

    void Matrix::set(
        std::size_t row,
        std::size_t col,
        double value
    ) noexcept
    {
        data_[row * cols_ + col] = value;
    }

    Matrix& Matrix::operator+=(const Matrix& other)
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument(
                "Matrix dimensions must match for operator+="
            );
        }

        for (std::size_t i{0}; i < data_.size(); ++i)
        {
            data_[i] += other.data_[i];
        }

        return *this;
    }

    Matrix Matrix::operator+(const Matrix& other) const
    {
        Matrix result{*this};
        result += other;
        return result;
    }

    void Matrix::multiply_to(
        const Matrix& a,
        const Matrix& b,
        Matrix& destination
    )
    {
        if (a.cols_ != b.rows_)
        {
            throw std::invalid_argument(
                "Matrix multiplication dimension mismatch"
            );
        }

        if (
            destination.rows_ != a.rows_ ||
            destination.cols_ != b.cols_
        )
        {
            throw std::invalid_argument(
                "Matrix multiplication destination dimensions mismatch"
            );
        }

        std::fill(
            destination.data_.begin(),
            destination.data_.end(),
            0.0
        );

        // i-k-j is considerably more cache friendly than
        // the traditional i-j-k arrangement.
        for (std::size_t i{0}; i < a.rows_; ++i)
        {
            for (std::size_t k{0}; k < a.cols_; ++k)
            {
                const double a_value = a(i, k);

                const double* b_row =
                    &b(k, 0);

                double* output_row =
                    &destination(i, 0);

                for (std::size_t j{0}; j < b.cols_; ++j)
                {
                    output_row[j] +=
                        a_value * b_row[j];
                }
            }
        }
    }

    Matrix Matrix::operator*(const Matrix& other) const
    {
        Matrix result{
            rows_,
            other.cols_
        };

        multiply_to(
            *this,
            other,
            result
        );

        return result;
    }

    Matrix Matrix::dense(
        const Matrix& input,
        const Matrix& weights,
        const Matrix& bias
    )
    {
        if (input.cols_ != weights.rows_)
        {
            throw std::invalid_argument(
                "Dense layer input dimension mismatch"
            );
        }

        if (
            bias.rows_ != 1 ||
            bias.cols_ != weights.cols_
        )
        {
            throw std::invalid_argument(
                "Dense bias must be [1 x output_size]"
            );
        }

        Matrix output{
            input.rows_,
            weights.cols_
        };

        multiply_to(
            input,
            weights,
            output
        );

        // Broadcast bias across every batch row.
        for (std::size_t r{0}; r < output.rows_; ++r)
        {
            double* output_row = &output(r, 0);

            for (std::size_t c{0}; c < output.cols_; ++c)
            {
                output_row[c] += bias(0, c);
            }
        }

        return output;
    }

    void Matrix::map_inplace(
        const std::function<double(double)>& function
    )
    {
        for (double& value : data_)
        {
            value = function(value);
        }
    }

    Matrix Matrix::map(
        const std::function<double(double)>& function
    ) const
    {
        Matrix result{*this};
        result.map_inplace(function);
        return result;
    }

    void Matrix::randomize(
        double min,
        double max
    )
    {
        if (min >= max)
        {
            throw std::invalid_argument(
                "Randomization minimum must be less than maximum"
            );
        }

        static thread_local std::mt19937 engine{
            std::random_device{}()
        };

        std::uniform_real_distribution<double> distribution{
            min,
            max
        };

        for (double& value : data_)
        {
            value = distribution(engine);
        }
    }

    void Matrix::print() const
    {
        for (std::size_t r{0}; r < rows_; ++r)
        {
            std::cout << "[ ";

            for (std::size_t c{0}; c < cols_; ++c)
            {
                std::cout << (*this)(r, c) << ' ';
            }

            std::cout << "]\n";
        }

        std::cout << '\n';
    }
}
