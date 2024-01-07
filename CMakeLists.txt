cmake_minimum_required(VERSION 3.20)

# basic settings 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# add main targets 
project(clothpose LANGUAGES C CXX)
# add_executable(zema)
add_library(clothpose)
target_include_directories(clothpose PUBLIC include)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp include/clothpose/*.hpp)
target_sources(clothpose PUBLIC ${sources})

# add igl 
add_subdirectory(extern/eigen)
add_subdirectory(extern/libigl)
add_subdirectory(extern/pybind11)
igl_include(glfw)

# handle link dependencies
target_link_libraries(clothpose PUBLIC igl::glfw Eigen3::Eigen)
# need to install libtbb-dev
find_package(TBB REQUIRED)
target_link_libraries(clothpose PRIVATE TBB::tbb)

pybind11_add_module(py_clothpose ${sources} pybind/pybind.cpp)
target_include_directories(py_clothpose PUBLIC include)
target_link_libraries(py_clothpose PRIVATE TBB::tbb igl::glfw Eigen3::Eigen)

# add test 
add_executable(test_io)
target_sources(test_io PUBLIC test/test_io.cpp)
target_link_libraries(test_io PUBLIC clothpose igl::glfw)

add_executable(test_optimizer)
target_sources(test_optimizer PUBLIC test/test_optimizer.cpp)
target_include_directories(test_optimizer PRIVATE extern/rapidjson/include)
target_link_libraries(test_optimizer PUBLIC clothpose igl::glfw)

# handle test resources 
FILE(GLOB resource_files RELATIVE ${CMAKE_SOURCE_DIR}/resources ${CMAKE_SOURCE_DIR}/resources/**/*)
FOREACH (rel_path ${resource_files})
    configure_file("${CMAKE_SOURCE_DIR}/resources/${rel_path}" "${CMAKE_CURRENT_BINARY_DIR}/resources/${rel_path}" COPYONLY)
ENDFOREACH()
