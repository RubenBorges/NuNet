# Source and header setups
set(SOURCES
    Matrix.cpp
    DenseLayer.cpp
    Tensor.cpp
    Conv2D.cpp
    CNN.cpp
)
set(HEADERPATHS
    include/Matrix.hpp
    include/DenseLayer.hpp
    include/Tensor.hpp
    include/Conv2D.hpp
    include/CNN.hpp

)

# add_executable(NuNet NuNet.cpp
#     ${SOURCES}
#     ${HEADERPATHS}
# )

add_executable(NuNet ConvNuNet.cpp 
    ${SOURCES} 
    ${HEADERPATHS}
)

# =====================================================================
# SYSTEM PACMAN LLAMA LINKING RULES (Arch Linux)
# =====================================================================

# 1. Look up the pacman system shared library file natively inside /usr/lib
# find_library(LLAMA_LIB
#     NAMES llama
#     PATHS /usr/lib
#     NO_DEFAULT_PATH
#     REQUIRED
# )

# 2. Add search parameters for your local project headers and system-wide headers
target_include_directories(NuNet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)
find_package(TBB REQUIRED COMPONENTS tbb)
target_link_libraries(NuNet PRIVATE TBB::tbb ${OpenCV_LIBS})

if(NUNET_ENABLE_SYCL)
    include(CheckCXXCompilerFlag)
    include(CheckIncludeFileCXX)

    check_cxx_compiler_flag("-fsycl" NUNET_COMPILER_HAS_FSYCL)
    if(NOT NUNET_COMPILER_HAS_FSYCL)
        message(FATAL_ERROR "NUNET_ENABLE_SYCL requires a compiler supporting -fsycl")
    endif()

    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    check_include_file_cxx("sycl/sycl.hpp" NUNET_HAS_SYCL_HEADER)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT NUNET_HAS_SYCL_HEADER)
        message(FATAL_ERROR "NUNET_ENABLE_SYCL requires sycl/sycl.hpp")
    endif()

    target_compile_definitions(NuNet PRIVATE NUNET_ENABLE_SYCL=1)
    target_compile_options(NuNet PRIVATE -fsycl)
    target_link_options(NuNet PRIVATE -fsycl)
endif()

# Compiler optimizations
target_compile_options(NuNet PRIVATE -O3 -Wall -Wextra -march=native)
