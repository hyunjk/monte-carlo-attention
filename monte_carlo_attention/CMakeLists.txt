cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(NAME monte-carlo-attention)
set(CMAKE_PREFIX_PATH "~/Workspace/libtorch/share/cmake/Torch")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
set(PYTHON_EXECUTABLE "/usr/bin/python3")

project(${NAME})

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 14)

cuda_add_executable(${NAME} test.cpp test.h attention.cu attention.cuh)
target_link_libraries(${NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 14)
