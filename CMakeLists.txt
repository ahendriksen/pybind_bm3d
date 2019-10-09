cmake_minimum_required(VERSION 2.8.12)
project(pybind_bm3d)

find_package (bm3d REQUIRED)
find_package (pybind11 REQUIRED)

pybind11_add_module(pybind_bm3d src/main.cpp)
target_link_libraries (pybind_bm3d PRIVATE bm3d )
target_include_directories(pybind_bm3d PRIVATE bm3d)
