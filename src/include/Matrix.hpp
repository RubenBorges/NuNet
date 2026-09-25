#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <compare>
#include <random>
namespace bpy {

class Matrix {
private:
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data;

public:
    Matrix(std::size_t rows, std::size_t cols, double initValue = 0.0);
    ~Matrix() = default;

    // Element Access
    double&       operator()  (std::size_t r, std::size_t c);
    const double& operator()  (std::size_t r, std::size_t c) const;

    // Compound Operators (In-place, Zero-allocation)
    Matrix& operator+=(const Matrix& other);

    // Standard Binary Operators (Leverage compound operators internally)
    Matrix operator*(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const {
        Matrix result = *this;
        result += other;
        return result;
    }
    
    auto operator<=> (const Matrix& other) const = default;
    
    void randomize(double min = -1.0, double max = 1.0);
    
    // In-place structural mutations
    void map_inplace(const std::function<double(double)>& func);
    Matrix map(const std::function<double(double)>& func) const;

    // High Performance Dense evaluation (Bypasses intermediate heap generation)
    static void multiply_to(const Matrix& A, const Matrix& B, Matrix& C);
    static Matrix dense(const Matrix& input, const Matrix& weights, const Matrix& bias);
    
    std::size_t Rows() const { return rows; }
    std::size_t Cols() const { return cols; }


    double get(int r, int c) const;
    void set(int r, int c, double val);

    void print() const;

    // Helper to fill a matrix with random weights between -0.5 and 0.5
    static void randomize_matrix(Matrix& m) {
        for (auto& val : m.data) {
            val = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
};

} // namespace bpy

