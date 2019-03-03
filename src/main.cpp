#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include <CL/cl.hpp>
#include <MatrixMultiplicationOpenCL/matrix.h>

int main(int argc, char **argv) 
{    
    Matrix A(3, 5);
    Matrix B(8, 3);

    for (size_t i = 0; i < A.data.size(); ++i)
        A.data[i] = 1;

    for (size_t i = 0; i < B.data.size(); ++i)
        B.data[i] = 1;

    if (!canMultiply(A, B))
        return -1;

    Matrix C(B.width, A.height);

    try 
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        auto context = cl::Context(CL_DEVICE_TYPE_GPU);

        auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

        auto commandQueue = cl::CommandQueue(context, devices[0]);


        // Read source file
        auto sourceFile = std::ifstream("../../kernels/naive_mult.cl");
        auto sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile),
                                      std::istreambuf_iterator<char>());

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

        // Make program of the source code in the context
        auto program = cl::Program(context, source);

        program.build(devices);

        auto kernel = cl::Kernel(program, "multiply");

        // Create memory buffers
        auto bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, A.data.size() * sizeof(float));
        auto bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, B.data.size() * sizeof(float));
        auto bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, C.data.size() * sizeof(float));

        // Copy lists A and B to the memory buffers
        commandQueue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, A.data.size() * sizeof(float), A.data.data());
        commandQueue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, B.data.size() * sizeof(float), B.data.data());

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

        auto r = commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, C.data.size() * sizeof(float), C.data.data());

        printMatrix(A);
        printMatrix(B);
        printMatrix(C);
        printMatrix( multiply(A, B) );

        auto time = end - start;
        std::cout << "time: " << std::chrono::duration<double>(time).count() << std::endl;
    }
    catch (cl::Error error) 
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        return -1;
    }

    return 0;
}