#include <iostream>
#include <MatrixMultiplicationOpenCL/matrix.h>
#include <random>

Matrix::Matrix(size_t w, size_t h, bool randomInit)
    : width(w), height(h), data(w * h)
{
    if (randomInit)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1000, 1000);

        for (auto& elem : data)
            elem = dis(gen);
    }
}

Matrix multiply(const Matrix& A, const Matrix& B)
{
    if (!canMultiply(A, B))
        return Matrix(0, 0);

    Matrix C(B.width, A.height);

    for (size_t y = 0; y < C.height; ++y)
    {
        for (size_t x = 0; x < C.width; ++x)
        {
            for (size_t i = 0; i < A.width; ++i)
            {
                C.data[C.width * y + x] += A.data[A.width * y + i] * B.data[B.width * i + x];
            }
        }
    }

    return C;
}

float sqrtNorm(const Matrix& A, const Matrix& B)
{
    if ( A.width != B.width || A.height != B.height )
        return -1;

    float sum = 0.0;

    for (size_t y = 0; y < A.height; ++y)
    {
        for (size_t x = 0; x < A.width; ++x)
        {
            auto index = A.width * y + x;
            auto sqr = (A.data[index] - B.data[index]) * (A.data[index] - B.data[index]);
            sum += sqrt(sqr);
        }
    }

    return sum / A.data.size();
}

void printMatrix(const Matrix& mat)
{
    for (auto iter = mat.data.begin(); iter != mat.data.end(); ++iter)
    {
        if ((iter - mat.data.begin()) % mat.width == 0)
            std::cout << std::endl;

        std::cout << *iter << " ";
    }

    std::cout << std::endl;
}

bool canMultiply(const Matrix& A, const Matrix& B)
{
    return A.width == B.height;
}

size_t operationsCount(const Matrix& A, const Matrix& B)
{
    if (!canMultiply(A, B))
        return 0;

    auto m = A.height;
    auto n = B.width;
    auto p = A.width;

    return (2 * p - 1) * m * n;
}