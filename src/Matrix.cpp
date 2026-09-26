#include <Matrix.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace bpy {

Matrix::Matrix(std::size_t rows, std::size_t cols, double init_value)
    : rows_{rows}, cols_{cols}, data_(rows * cols, init_value) {
  if (rows == 0 || cols == 0)
    throw std::invalid_argument("Matrix dimensions must be greater than zero");
}

double &Matrix::operator()(std::size_t row, std::size_t col) noexcept {
  return data_[row * cols_ + col];
}

const double &Matrix::operator()(std::size_t row,
                                 std::size_t col) const noexcept {
  return data_[row * cols_ + col];
}

Matrix &Matrix::operator+=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions must match");
  }

  for (std::size_t i = 0; i < data_.size(); ++i)
    data_[i] += other.data_[i];

  return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix result{*this};
  result += other;
  return result;
}

void Matrix::multiply_to(const Matrix &A, const Matrix &B, Matrix &C) {
  if (A.cols_ != B.rows_ || C.rows_ != A.rows_ || C.cols_ != B.cols_) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for multiplication");
  }

  std::fill(C.data_.begin(), C.data_.end(), 0.0);

  /*
      i-k-j ordering.

      This is generally faster than i-j-k for row-major
      matrices because B[k][j] and C[i][j] are contiguous.
  */

  for (std::size_t i = 0; i < A.rows_; ++i) {
    const std::size_t a_base = i * A.cols_;
    const std::size_t c_base = i * C.cols_;

    for (std::size_t k = 0; k < A.cols_; ++k) {
      const double a = A.data_[a_base + k];

      const std::size_t b_base = k * B.cols_;

      for (std::size_t j = 0; j < B.cols_; ++j) {
        C.data_[c_base + j] += a * B.data_[b_base + j];
      }
    }
  }
}

Matrix Matrix::operator*(const Matrix &other) const {
  Matrix result{rows_, other.cols_};

  multiply_to(*this, other, result);

  return result;
}

void Matrix::map_inplace(const std::function<double(double)> &function) {
  for (double &value : data_)
    value = function(value);
}

Matrix Matrix::map(const std::function<double(double)> &function) const {
  Matrix result{*this};
  result.map_inplace(function);
  return result;
}

Matrix Matrix::dense(const Matrix &input, const Matrix &weights,
                     const Matrix &bias) {
  if (input.cols_ != weights.rows_)
    throw std::invalid_argument("Dense input/weight dimensions do not match");

  if (bias.rows_ != 1 || bias.cols_ != weights.cols_) {
    throw std::invalid_argument("Dense bias dimensions do not match");
  }

  Matrix result{input.rows_, weights.cols_};

  multiply_to(input, weights, result);

  for (std::size_t r = 0; r < result.rows_; ++r) {
    const std::size_t base = r * result.cols_;

    for (std::size_t c = 0; c < result.cols_; ++c) {
      result.data_[base + c] += bias.data_[c];
    }
  }

  return result;
}

void Matrix::randomize(double min, double max) {
  if (!(min < max))
    throw std::invalid_argument(
        "Randomization minimum must be less than maximum");

  static thread_local std::mt19937_64 engine{std::random_device{}()};

  std::uniform_real_distribution<double> distribution{min, max};

  for (double &value : data_)
    value = distribution(engine);
}

Matrix Matrix::relu() const {
  Matrix result{*this};

  for (double &value : result.data_)
    value = std::max(0.0, value);

  return result;
}

Matrix Matrix::max_pool(std::size_t pool_size, std::size_t stride) const {
  if (pool_size == 0 || stride == 0)
    throw std::invalid_argument(
        "Pool size and stride must be greater than zero");

  if (pool_size > rows_ || pool_size > cols_) {
    throw std::invalid_argument("Pool window is larger than matrix");
  }

  const std::size_t output_rows = (rows_ - pool_size) / stride + 1;

  const std::size_t output_cols = (cols_ - pool_size) / stride + 1;

  Matrix result{output_rows, output_cols};

  for (std::size_t r = 0; r < output_rows; ++r) {
    for (std::size_t c = 0; c < output_cols; ++c) {
      double maximum = -std::numeric_limits<double>::infinity();

      for (std::size_t pr = 0; pr < pool_size; ++pr) {
        for (std::size_t pc = 0; pc < pool_size; ++pc) {
          maximum =
              std::max(maximum, (*this)(r * stride + pr, c * stride + pc));
        }
      }

      result(r, c) = maximum;
    }
  }

  return result;
}

Matrix Matrix::flatten() const {
  Matrix result{data_.size(), 1};

  std::copy(data_.begin(), data_.end(), result.data_.begin());

  return result;
}

Matrix Matrix::convolve(const Matrix &kernel, std::size_t stride) const {
  if (kernel.rows_ > rows_ || kernel.cols_ > cols_) {
    throw std::invalid_argument("Kernel cannot be larger than matrix");
  }

  if (stride == 0)
    throw std::invalid_argument("Stride must be greater than zero");

  const std::size_t output_rows = (rows_ - kernel.rows_) / stride + 1;

  const std::size_t output_cols = (cols_ - kernel.cols_) / stride + 1;

  Matrix output{output_rows, output_cols};

  for (std::size_t r = 0; r < output_rows; ++r) {
    for (std::size_t c = 0; c < output_cols; ++c) {
      double sum = 0.0;

      for (std::size_t kr = 0; kr < kernel.rows_; ++kr) {
        const std::size_t input_base = (r * stride + kr) * cols_;

        const std::size_t kernel_base = kr * kernel.cols_;

        for (std::size_t kc = 0; kc < kernel.cols_; ++kc) {
          sum += data_[input_base + c * stride + kc] *
                 kernel.data_[kernel_base + kc];
        }
      }

      output(r, c) = sum;
    }
  }

  return output;
}

double Matrix::get(std::size_t row, std::size_t col) const noexcept {
  return (*this)(row, col);
}

void Matrix::set(std::size_t row, std::size_t col, double value) noexcept {
  (*this)(row, col) = value;
}

void Matrix::print() const {
  for (std::size_t r = 0; r < rows_; ++r) {
    std::cout << "[ ";

    for (std::size_t c = 0; c < cols_; ++c)
      std::cout << (*this)(r, c) << ' ';

    std::cout << "]\n";
  }

  std::cout << '\n';
}

} // namespace bpy
