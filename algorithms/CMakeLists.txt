cmake_minimum_required(VERSION 3.12)
project(algorithms)

set(CMAKE_CXX_STANDARD 17)

# output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCE_EXE main.cpp)
set(TOP_DIR ${CMAKE_SOURCE_DIR})
set(SRC_DIR ${TOP_DIR}/src)

include_directories(${CMAKE_SOURCE_DIR}/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/stats)
include_directories(${CMAKE_SOURCE_DIR}/gcem)
include_directories(${SRC_DIR})

set(SRC_LIST ${SRC_DIR}/Sample.cpp
			   ${SRC_DIR}/RegressionModel.cpp
			   )
add_library(Sample STATIC ${SRC_LIST})
add_library(RegressionModel STATIC ${SRC_LIST})
set_target_properties(Sample RegressionModel PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Sample)
target_link_libraries(main PRIVATE RegressionModel)
