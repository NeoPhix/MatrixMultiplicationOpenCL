cmake_minimum_required(VERSION 3.1)
project(MatrixMultiplicationOpenCL)

add_definitions(-std=c++14)

find_package(OpenCL REQUIRED)

include_directories(include ${OpenCL_INCLUDE_DIRS})

add_executable(main src/main.cpp 
	                src/matrix.cpp 
	                include/MatrixMultiplicationOpenCL/matrix.h)

target_link_libraries(main ${OpenCL_LIBRARY})



