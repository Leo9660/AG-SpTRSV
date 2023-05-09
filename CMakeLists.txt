cmake_minimum_required(VERSION 3.5)

#set(CMAKE_C_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(SpTRSV LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SRC_FILE ${SRC_DIR}/common.cpp
${SRC_DIR}/GPU_setup.cpp
${SRC_DIR}/schedule.cu
${SRC_DIR}/preprocessing.cu
${SRC_DIR}/finalize.cu
${SRC_DIR}/SpTRSV_executor.cu
${SRC_DIR}/specialization.cpp
${SRC_DIR}/format_def.cpp
${SRC_DIR}/transformation.cu
${SRC_DIR}/elimination.cpp
)

set(UTILS_DIR ${PROJECT_SOURCE_DIR}/utils)
set(UTILS_FILE ${UTILS_DIR}/utils.cpp)

cuda_add_library(SpTRSV ${SRC_FILE})
add_library(utils ${UTILS_FILE})

#add_library(${PROJECT_NAME} STATIC)
#target_link_libraries(SpTRSV)