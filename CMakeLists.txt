cmake_minimum_required(VERSION 3.22)

project(better_loop)

set(CMAKE_CXX_STANDARD 17)
find_package(PCL 1.10 REQUIRED COMPONENTS common io registration filters)
find_package(Ceres REQUIRED)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB SOURCES "verify/*.cpp")

add_executable(better_loop ${SOURCES} main.cpp)
target_link_libraries(better_loop 
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
)

add_executable(eval eval.cpp)
