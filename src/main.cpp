#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <CL/cl.hpp>

template <class T>
void printMatrix(const std::vector<T>& data, size_t width, size_t height)
{
	if (data.size() != width * height)
	{
		std::cout << "Error! Data size is mismatched with matrix width and height." << std::endl;
		return;
	}

	for (auto iter = data.begin(); iter != data.end(); ++iter)
	{
		if ((iter - data.begin()) % width == 0)
			std::cout << std::endl;

		std::cout << *iter << " ";
	}

	std::cout << std::endl;
}


int main(int argc, char **argv) 
{
	constexpr size_t width = 10;
	constexpr size_t height = 5;
	constexpr size_t N = width * height;
	std::vector<double> A(N);
	std::vector<double> B(N);
	std::vector<double> C(N);

	for (auto i = 0; i < N; ++i)
	{
		A[i] = i;
		B[i] = i * 3;
	}

	std::vector<cl::Platform> platforms;
	auto ret = cl::Platform::get(&platforms);

	auto context = cl::Context(CL_DEVICE_TYPE_GPU);

	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	auto commandQueue = cl::CommandQueue(context, devices[0]);


	// Read source file
	auto sourceFile = std::ifstream("../../kernels/test_kernel.cl");
	auto sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile),
								  std::istreambuf_iterator<char>());

	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

	// Make program of the source code in the context
	auto program = cl::Program(context, source);

	program.build(devices);

	auto kernel = cl::Kernel(program, "matrix_sum");

	// Create memory buffers
	auto bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N * sizeof(double));
	auto bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N * sizeof(double));
	auto bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double));

	// Copy lists A and B to the memory buffers
	commandQueue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N * sizeof(double), A.data());
	commandQueue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N * sizeof(double), B.data());

	// Set arguments to kernel
	kernel.setArg(0, bufferA);
	kernel.setArg(1, bufferB);
	kernel.setArg(2, bufferC);

	// Run the kernel on specific ND range
	cl::NDRange global(N);
	cl::NDRange local(1);
	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

	// Read buffer C into a local list
	commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N * sizeof(double), C.data());

	printMatrix(A, width, height);
	printMatrix(B, width, height);
	printMatrix(C, width, height);
}