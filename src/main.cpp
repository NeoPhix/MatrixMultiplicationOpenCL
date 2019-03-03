#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

#include <CL/cl.hpp>
#include <MatrixMultiplicationOpenCL/matrix.h>

struct GPUInfo
{
    std::string name;
    size_t coresCount;
    double MHz;
};

void showResults(const std::chrono::duration<long long, std::nano> time, 
                 size_t operationsCount, double norm, const GPUInfo& info);

bool initFromFile(const char* filename, std::string& kernelAddress, 
                  GPUInfo& info, Matrix& A, Matrix& B);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "No parameters file. Need add path to file with parameters to arguments" << std::endl;
        return 1;
    }


    std::string kernelAddress;
    GPUInfo info;
    Matrix A;
    Matrix B;
    
    if (!initFromFile(argv[1], kernelAddress, info, A, B))
        return 1;

    Matrix C(B.width, A.height);
    Matrix D = multiply(A, B);

    try 
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        auto context = cl::Context(CL_DEVICE_TYPE_GPU);
        auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
        auto commandQueue = cl::CommandQueue(context, devices[0]);

        // Read source file
        auto sourceFile = std::ifstream(kernelAddress);

        if (!sourceFile.is_open())
        {
            std::cout << "Cannot open kernel file " << kernelAddress << std::endl;
            return 1;
        }

        auto sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile),
                                      std::istreambuf_iterator<char>());

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

        // Make program of the source code in the context
        auto program = cl::Program(context, source);
        program.build(devices);
        auto kernel = cl::Kernel(program, "multiply");

        // Create memory buffers
        auto bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, A.data.size() * sizeof(double));
        auto bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, B.data.size() * sizeof(double));
        auto bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, C.data.size() * sizeof(double));

        // Copy lists A and B to the memory buffers
        commandQueue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, A.data.size() * sizeof(double), A.data.data());
        commandQueue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, B.data.size() * sizeof(double), B.data.data());

        // Set arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, A.width);
        kernel.setArg(4, B.width);

        // Run the kernel on specific ND range
        auto global = cl::NDRange(C.width, C.height);
        auto local = cl::NDRange(1, 1);

        auto start = std::chrono::steady_clock::now();
        commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        auto end = std::chrono::steady_clock::now();

        commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, C.data.size() * sizeof(double), C.data.data());
        showResults(end - start, operationsCount(A, B), sqrtNorm(C, D), info);

    }
    catch (cl::Error error) 
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}


bool initFromFile(const char* filename, std::string& kernelAddress, GPUInfo& info, Matrix& A, Matrix& B)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cout << "Cannot open file " << filename << std::endl;
        return false;
    }

    std::getline(file, kernelAddress);

    std::getline(file, info.name);
    file >> info.coresCount;
    file >> info.MHz;

    size_t widthA;
    size_t heightA;
    size_t widthB;
    size_t heightB;

    file >> widthA;
    file >> heightA;
    file >> widthB;
    file >> heightB;

    A = Matrix(widthA, heightA, true);
    B = Matrix(widthB, heightB, true);

    if (!canMultiply(A, B))
    {
        std::cout << "Input matrices cannot be multiplyed because of mismatched size" << std::endl;
        return false;
    }

    return true;
}



void showResults(const std::chrono::duration<long long, std::nano> time,
    size_t operationsCount,
    double norm,
    const GPUInfo& info)
{
    auto peac = info.coresCount * info.MHz / 1000;
    std::cout << "Video Card: " << info.name << std::endl
        << "Cores: " << info.coresCount << std::endl
        << "MHz: " << info.MHz << std::endl
        << "Performance peac (GFLOP/s): " << peac << std::endl << std::endl;

    constexpr double epsilon = 0.000001;
    std::cout << "SQRT norm: " << norm << ". ";
    if (norm < epsilon)
        std::cout << "Result is correct" << std::endl;
    else
        std::cout << "Error! Result is not correct! Norm is greater than epsilon = " << epsilon << std::endl;

    auto msCount = std::chrono::duration<double, std::milli>(time).count();
    auto nsCount = std::chrono::duration<double, std::nano>(time).count();
    auto performance = operationsCount / nsCount;

    std::cout << "Multiplication time: " << msCount << " ms" << std::endl;
    std::cout << "Performance: " << operationsCount / nsCount << " GFLOP/s" << std::endl;
    std::cout << "Theoretical peac: " << performance / peac * 100 << " % of peac" << std::endl;
}