#include <iostream>
#include <CL/cl.h>
  
int main(int argc, char **argv) 
{
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint num_devices;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform_id, &num_platforms);
    cl_int ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  
    std::cout << "platform_id: " << platform_id << std::endl
              << "device_id: " << device_id << std::endl
              << "num_devices: " << num_devices << std::endl
              << "num_platforms: " << num_platforms << std::endl;
}