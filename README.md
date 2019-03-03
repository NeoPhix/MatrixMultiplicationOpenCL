# MatrixMultiplicationOpenCL
A repo for tests of matrix multiplications using OpenCL technique

## How to use?

The code was tested on GPU nVidia GeForce GTX 750 with OpenCL 1.2 driver from CUDA 8.1 toolkit.
The code uses C++ OpenCL bindings (be sure that you use OpenCL with CL/cl.hpp file)

### Build 


```
git clone https://github.com/NeoPhix/MatrixMultiplicationOpenCL.git
cd MatrixMultiplicationOpenCL
mkdir build && cd build
cmake ..
```
After getting the project you need to build it by your own compiler.

### Run

```
main ../tests/test.txt
``` 

Be sure that you set right test filename.

### Test file structure

The input file has the following data:

```
../../kernels/naive_mult.cl  # Kernel code address
nVidia GeForce GTX 750       # GPU name
512                          # GPU cores count
1020                         # GPU cores MHz

500                          # First matrix width
500                          # First matrix height
500                          # Second matrix width
500                          # Second matrix height
```

The input matrices are filled by random float (double) numbers in range [-1000, 1000].

## Additional info

Thanks to authors of the following tutorials:
https://cnugteren.github.io/tutorial/pages/page1.html
https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/