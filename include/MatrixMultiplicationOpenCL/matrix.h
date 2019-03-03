#ifndef MATRIX_MULTIPLICATION_OPENCL_MATRIX_H
#define MATRIX_MULTIPLICATION_OPENCL_MATRIX_H

#include <vector>

struct Matrix
{
    Matrix() = delete;
    Matrix(size_t w, size_t h) : width(w), height(h), data(w * h) {}
    ~Matrix() = default;

    size_t width;
    size_t height;
    std::vector<float> data;
};

Matrix multiply(const Matrix& A, const Matrix& B);

void printMatrix(const Matrix& mat);
bool canMultiply(const Matrix& A, const Matrix& B);
size_t operationsCount(const Matrix& A, const Matrix& B);

#endif
