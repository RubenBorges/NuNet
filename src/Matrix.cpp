#include <random>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <Matrix.hpp>

namespace bpy {

Matrix::Matrix(std::size_t rows, std::size_t cols, double initValue): rows(rows), cols(cols), data(rows * cols, initValue) {}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(std::size_t r, std::size_t c) const {
    return data[r * cols + c];
}

// In-place addition: loops contiguously over flat memory without resizing
Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for operator+=");
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

// CACHE-OPTIMIZED multiplication engine (Row-K-Col loop nesting)
void Matrix::multiply_to(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        throw std::invalid_argument("Matrix dimension mismatch in multiply_to target initialization");
    }
    
    // Clear destination buffer
    std::fill(C.data.begin(), C.data.end(), 0.0);

    // Row-K-Col traversal allows the inner loop to stream contiguously 
    // across both Matrix B and Matrix C memory rows.
    for (std::size_t r = 0; r < A.rows; ++r) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            double a_val = A(r, k); // Cache the single value from A
            
            // Inner loop now iterates over columns linearly (Contiguous memory access)
            for (std::size_t c = 0; c < B.cols; ++c) {
                C(r, c) += a_val * B(k, c);
            }
        }
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result(rows, other.cols);
    multiply_to(*this, other, result);
    return result;
}

// Mutates values directly within the current instance, avoiding memory copying
void Matrix::map_inplace(const std::function<double(double)>& func) {
    for (auto& val : data) {
        val = func(val);
    }
}

Matrix Matrix::map(const std::function<double(double)>& func) const {
    Matrix result = *this;
    result.map_inplace(func);
    return result;
}

// HIGH-PERFORMANCE + BROADCAST AWARE Dense evaluation
Matrix Matrix::dense(const Matrix& input, const Matrix& weights, const Matrix& bias) {
    if (bias.Rows() != 1 || bias.Cols() != weights.Cols()) {
        throw std::invalid_argument("Bias must be a row vector matching weights' output columns [1 x weights.cols]");
    }

    Matrix destination(input.rows, weights.cols);
    
    // 1. Perform optimized matrix multiplication
    multiply_to(input, weights, destination);
    
    // 2. Broadcast the bias row to EVERY sample row in the batch matrix
    for (std::size_t r = 0; r < destination.rows; ++r) {
        for (std::size_t c = 0; c < destination.cols; ++c) {
            destination(r, c) += bias(0, c);
        }
    }
    
    return destination; // Returned via RVO
}

// Optimized randomize: Thread-safe, persisting engine via static thread_local
void Matrix::randomize(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("Min value must be less than max value.");
    }
    // Static thread_local avoids recreating the random engine infrastructure on every function call
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    for (auto& val : data) { 
        val = dis(gen); 
    }
}

void Matrix::print() const {
    for (std::size_t r = 0; r < rows; ++r) {
        std::cout << "[ ";
        for (std::size_t c = 0; c < cols; ++c) { std::cout << (*this)(r, c) << " "; }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

double Matrix::get(int r, int c) const { return data[r * cols + c]; }

void Matrix::set(int r, int c, double val) { data[r * cols + c] = val; }

} // namespace bpy
