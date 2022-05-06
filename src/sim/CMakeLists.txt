cmake_minimum_required(VERSION 3.16)

add_library(sim STATIC Simulator.cpp
                       Simulator.hpp)

target_link_libraries(sim PRIVATE kompute::kompute logger)