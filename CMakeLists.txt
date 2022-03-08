cmake_minimum_required(VERSION 3.10)
project(nvinfer_dcnv2_plugin LANGUAGES CUDA CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall") 

find_package(CUDA 11 REQUIRED)
find_library(nvinfer REQUIRED)

# set(CMAKE_CUDA_STANDARD 17)
add_library(nvinfer_dcnv2_plugin SHARED
    src/DCNv2.cpp
    src/dcn_v2_im2col_cuda.cu)

target_include_directories(nvinfer_dcnv2_plugin PRIVATE
    src
    /usr/local/cuda/include)

set_target_properties(nvinfer_dcnv2_plugin PROPERTIES
    CXX_STANDARD "17"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO")

target_link_libraries(nvinfer_dcnv2_plugin
    nvinfer
    cuda
    cudart
    cublas
    nvinfer_plugin
    nvparsers)
