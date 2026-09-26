cmake_minimum_required(VERSION 3.20)
project(NuNet CXX)

# Force C++26 for modern features
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUNET_ENABLE_SYCL "Build optional SYCL GPU execution kernels" OFF)

# Process the source subdirectory first
add_subdirectory(src)

# =====================================================================
# INSTALL & DISTRIBUTION TARGETS (dist/)
# =====================================================================

set(DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist")

# 1. Install the compiled binary executable
install(TARGETS NuNet
    RUNTIME DESTINATION "${DIST_DIR}"
)

# 2. Copy the text documentation assets folder contents
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/model/"
    DESTINATION "${DIST_DIR}/model"
    FILES_MATCHING PATTERN "*.txt"
)

