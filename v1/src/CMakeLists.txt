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

# Compiler optimizations
target_compile_options(NuNet PRIVATE -O3 -Wall -Wextra -march=native)
