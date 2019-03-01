#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <CL/cl.hpp>

template <class T>
struct Matrix
{
	Matrix() = delete;
	Matrix(size_t w, size_t h) : width(w), height(h), data(w * h) {}
	~Matrix() = default;
	size_t width;
	size_t height;
	std::vector<T> data;
};

template <class T>
void printMatrix(const Matrix<T>& mat)
{
	for (auto iter = mat.data.begin(); iter != mat.data.end(); ++iter)
	{
		if ((iter - mat.data.begin()) % mat.width == 0)
			std::cout << std::endl;

		std::cout << *iter << " ";
	}

	std::cout << std::endl;
}


int main(int argc, char **argv) 
{
	Matrix<double> A(1, 5);
	Matrix<double> B(5, 1);
	Matrix<double> C(A.height, B.width);

	for (size_t i = 0; i < A.data.size(); ++i)
		A.data[i] = i;

	for (size_t i = 0; i < B.data.size(); ++i)
		B.data[i] = i;

	try 
	{
		std::vector<cl::Platform> platforms;
		auto ret = cl::Platform::get(&platforms);

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
		commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

		commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, C.data.size() * sizeof(double), C.data.data());

		printMatrix(A);
		printMatrix(B);
		printMatrix(C);
	}
	catch (cl::Error error) 
	{
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}