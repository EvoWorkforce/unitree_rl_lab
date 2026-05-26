cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# 1. Find packages
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# 2. Handle ONNX Architecture Dynamically
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNX_ARCH "linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(ONNX_ARCH "linux-aarch64-1.22.0")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ONNX_ROOT_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-${ONNX_ARCH}")
set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/libonnxruntime.so.1.22.0")

# 3. Define the library target
file(GLOB_RECURSE ADD_SRC_LIST "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${PROJECT_NAME}_lib SHARED ${ADD_SRC_LIST})

# Target-specific includes for the library
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${ONNX_ROOT_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

# Target-specific linking for the library
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
  unitree_sdk2 ddsc ddscxx rt pthread
  Boost::program_options
  yaml-cpp
  fmt
  ${ONNX_LIB}
)

# 4. Define the executable target
add_executable(g1_ctrl main.cpp)

# The executable automatically inherits includes and libraries from the library via PUBLIC linking
target_link_libraries(g1_ctrl PRIVATE ${PROJECT_NAME}_lib)
