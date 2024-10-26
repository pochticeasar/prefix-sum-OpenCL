cmake_minimum_required(VERSION 3.16)
project(gpu24-backlog-ocl-prefsum-pochticeasar C)
SET(CMAKE_C_STANDARD 11)
find_package(OpenCL REQUIRED)

set(CL_TARGET_OPENCL_VERSION "100")

add_executable(gpu24-backlog-ocl-matmul-pochticeasar main.c gpu_realization.c)
target_link_libraries(gpu24-backlog-ocl-matmul-pochticeasar OpenCL::OpenCL)
target_include_directories (gpu24-backlog-ocl-matmul-pochticeasar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
