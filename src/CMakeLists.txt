# Source and header setups
set(NNT_SOURCES
    NuNet/Matrix.cpp
    NuNet/DenseLayer.cpp
    NuNet/Tensor.cpp
    NuNet/Conv2D.cpp
    NuNet/CNN.cpp
)

set(ODNNT_SOURCES
    OneDNNet/OneDNNet.cpp
)

set(HEADERPATHS
    include/Matrix.hpp
    include/DenseLayer.hpp
    include/Tensor.hpp
    include/Conv2D.hpp
)

# Initialize Binary Executable Target
add_executable(NuNet ${ODNNT_SOURCES})

# Include directory lookup setup
target_include_directories(NuNet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Standard dependencies lookup
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)
find_package(TBB REQUIRED COMPONENTS tbb)

# Robust lower-case oneDNN finding wrapper
if(NOT TARGET DNNL::dnnl)
    find_package(dnnl QUIET)
    if(NOT dnnl_FOUND)
        find_package(oneDNN REQUIRED)
    endif()
endif()

# Link dependencies using standard lowercase DNNL::dnnl target mapping
target_link_libraries(NuNet PRIVATE TBB::tbb DNNL::dnnl ${OpenCV_LIBS})

# Global Compiler options 
target_compile_options(NuNet PRIVATE -O3 -Wall -Wextra -march=native)

# Explicitly bind SYCL features to compile OneDNNet.cpp
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_definitions(NuNet PRIVATE NUNET_ENABLE_SYCL=1)
    target_compile_options(NuNet PRIVATE -fsycl)
    target_link_options(NuNet PRIVATE -fsycl)
    message(STATUS "Intel SYCL compilation flags explicitly bound to NuNet")
else()
    message(FATAL_ERROR "You must configure CMake with icpx to compile this SYCL program.")
endif()
