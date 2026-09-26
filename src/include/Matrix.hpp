#pragma once

#include <Execution.hpp>
#include <opencv2/opencv.hpp>
#include <cstddef>
#include <functional>
#include <vector>
#include <filesystem>

namespace bpy {

class Matrix {
private:
    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<double> data_;

public:
    Matrix(std::size_t rows, std::size_t cols, double init_value = 0.0);

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;
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
        const std::function<double(double)>& function,
        ExecutionPolicy policy = ExecutionPolicy::CPU);

    Matrix map(
        const std::function<double(double)>& function) const;

    static void multiply_to(
        const Matrix& A,
        const Matrix& B,
        Matrix& C,
        ExecutionPolicy policy = ExecutionPolicy::CPU);

    static Matrix dense(
        const Matrix& input,
        const Matrix& weights,
        const Matrix& bias,
        ExecutionPolicy policy = ExecutionPolicy::CPU);

    void randomize(double min = -1.0, double max = 1.0);
    
static Matrix from_image(const std::filesystem::path& path, bool normalize = true) {

    cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) throw std::invalid_argument("Error: Image file is empty or could not be read!");
    
    Matrix mat(img.rows, img.cols);
    
    cv::Mat dest_view(img.rows, img.cols, CV_64F, mat.Data());

    double scale = normalize ? (1.0 / 255.0) : 1.0;
    img.convertTo(dest_view, CV_64F, scale); 

    return mat; 
}


    [[nodiscard]]
    Matrix relu() const;

    [[nodiscard]]
    Matrix max_pool(
        std::size_t pool_size = 2,
        std::size_t stride = 2) const;

    [[nodiscard]]
    Matrix flatten() const;

    [[nodiscard]]
    Matrix convolve(
        const Matrix& kernel,
        std::size_t stride = 1) const;

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

    [[nodiscard]]
    double* Data() noexcept {
        return data_.data();
    }

    [[nodiscard]]
    const double* Data() const noexcept {
        return data_.data();
    }

    double get(
        std::size_t row,
        std::size_t col) const noexcept;

    void set(
        std::size_t row,
        std::size_t col,
        double value) noexcept;

    void print() const;
};

} // namespace bpy
