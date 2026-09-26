#pragma once

#include <Execution.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>

namespace bpy {


struct imagePath {
    std::filesystem::path path;

    imagePath(const char* pathStr) : path(pathStr) {}
    imagePath(std::string pathString) : path(std::move(pathString)) {}
    imagePath(std::filesystem::path p) : path(std::move(p)) {}
};


class Tensor3D {
private:
    std::size_t channels_{};
    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<double> data_;

    [[nodiscard]]
    std::size_t index(
        std::size_t channel,
        std::size_t row,
        std::size_t col) const noexcept
    {
        return (channel * rows_ + row) * cols_ + col;
    }

public:
    Tensor3D(std::size_t channels, std::size_t rows, std::size_t cols, double init_value = 0.0);

    /** @brief Constructs a Tensor3D directly from a 1-channel image file.**/
    Tensor3D(const imagePath& imgPath, bool normalize = true);

    Tensor3D(const Tensor3D&) = default;
    Tensor3D(Tensor3D&&) noexcept = default;
    Tensor3D& operator=(const Tensor3D&) = default;
    Tensor3D& operator=(Tensor3D&&) noexcept = default;
    ~Tensor3D() = default;

    double& operator()(
        std::size_t channel,
        std::size_t row,
        std::size_t col) noexcept;

    const double& operator()(
        std::size_t channel,
        std::size_t row,
        std::size_t col) const noexcept;


    double& operator()(
        std::size_t row,
        std::size_t col) noexcept;

    const double& operator()(
        std::size_t row,
        std::size_t col) const noexcept;
        
    [[nodiscard]]
    std::size_t Channels() const noexcept { return channels_; }

    [[nodiscard]]
    std::size_t Rows() const noexcept { return rows_; }

    [[nodiscard]]
    std::size_t Cols() const noexcept { return cols_; }

    [[nodiscard]]
    std::size_t Size() const noexcept { return data_.size(); }

    [[nodiscard]]
    double* Data() noexcept { return data_.data(); }

    [[nodiscard]]
    const double* Data() const noexcept { return data_.data(); }

    void fill(double value) noexcept;

    void randomize(double min = -1.0, double max = 1.0);

    // Static factory function to construct a Tensor3D from a 1-channel image
    static Tensor3D from_image(const std::filesystem::path& path, bool normalize = true) {

        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        
        if (img.empty()) throw std::invalid_argument("Error: Image file is empty or could not be read: " + path.string());

        Tensor3D tensor(1, static_cast<std::size_t>(img.rows), static_cast<std::size_t>(img.cols));

        cv::Mat dest_view(img.rows, img.cols, CV_64F, tensor.Data());

        double scale = normalize ? (1.0 / 255.0) : 1.0;

        img.convertTo(dest_view, CV_64F, scale);

        return tensor; 
    }   

    [[nodiscard]]
    Tensor3D relu(
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Sender<Tensor3D> relu_async(
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Tensor3D max_pool(
        std::size_t pool_size,
        std::size_t stride,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Sender<Tensor3D> max_pool_async(
        std::size_t pool_size,
        std::size_t stride,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    std::vector<double> flatten(
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Sender<std::vector<double>> flatten_async(
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Tensor3D convolve(
        const Tensor3D& kernels,
        const std::vector<double>& biases,
        std::size_t stride = 1,
        std::size_t padding = 0,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;

    [[nodiscard]]
    Sender<Tensor3D> convolve_async(
        const Tensor3D& kernels,
        const std::vector<double>& biases,
        std::size_t stride = 1,
        std::size_t padding = 0,
        ExecutionPolicy policy = ExecutionPolicy::CPU) const;
        
    void print() const;
};

} // namespace bpy
