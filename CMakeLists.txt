cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SRC_FILES
    cxx/geometry_functions.cxx
    main.cxx
)

# Add executable
add_executable(main ${SRC_FILES})

# Include directories
target_include_directories(main PRIVATE
    /usr/local/include
    /usr/local/include/eigen3
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0
)

# Link libraries
target_link_libraries(main
    spdlog
)
