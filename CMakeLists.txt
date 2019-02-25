cmake_minimum_required(VERSION 3.1)
project(MatrixMultiplicationOpenCL)

add_definitions(-std=c++14)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

# Base controller node
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCL_LIBRARY})



