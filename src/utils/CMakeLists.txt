cmake_minimum_required(VERSION 3.16)

add_library(utils TickDurationHistory.cpp
                  TickRate.cpp
                  RNG.cpp)

target_link_libraries(utils PRIVATE fmt::fmt)
