#ifndef MATRIX_MULTIPLICATION_OPENCL_MATRIX_H
#define MATRIX_MULTIPLICATION_OPENCL_MATRIX_H

#include <vector>

struct Matrix
{
    Matrix() = delete;
    Matrix(size_t w, size_t h, bool randomInit = false); 
    ~Matrix() = default;

    size_t width;
    size_t height;
    std::vector<double> data;
};

Matrix multiply(const Matrix& A, const Matrix& B);
float sqrtNorm(const Matrix& A, const Matrix& B);

void printMatrix(const Matrix& mat);
bool canMultiply(const Matrix& A, const Matrix& B);
size_t operationsCount(const Matrix& A, const Matrix& B);

#endif
