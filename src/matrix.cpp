#include <iostream>
#include <MatrixMultiplicationOpenCL/matrix.h>


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