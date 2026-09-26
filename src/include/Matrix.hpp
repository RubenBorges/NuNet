#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace bpy {

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:
    Matrix(
        std::size_t rows,
        std::size_t cols,
        double init_value = 0.0);

    ~Matrix() = default;

    double& operator()(
        std::size_t row,
        std::size_t col) noexcept;

    const double& operator()(
        std::size_t row,
        std::size_t col) const noexcept;

    Matrix& operator+=(const Matrix& other);

    Matrix operator+(const Matrix& other) const;

    Matrix operator*(const Matrix& other) const;

    void map_inplace(
        const std::function<double(double)>& function);

    Matrix map(
        const std::function<double(double)>& function) const;

    static void multiply_to(
        const Matrix& A,
        const Matrix& B,
        Matrix& C);

    static Matrix dense(
        const Matrix& input,
        const Matrix& weights,
        const Matrix& bias);

    void randomize(
        double min = -1.0,
        double max = 1.0);

    [[nodiscard]]
    std::size_t Rows() const noexcept {
        return rows_;
    }

    [[nodiscard]]
    std::size_t Cols() const noexcept {
        return cols_;
    }

    [[nodiscard]]
    std::size_t Size() const noexcept {
        return data_.size();
    }

    double get(
        std::size_t row,
        std::size_t col) const noexcept;

    void set(
        std::size_t row,
        std::size_t col,
        double value) noexcept;

    void print() const;

    static void randomize_matrix(
        Matrix& matrix);
};

} // namespace bpy
