cmake_minimum_required(VERSION 3.16)

add_library(utils TickDurationHistory.cpp
                  TickRate.cpp
                  Timer.cpp
                  RNG.cpp)

target_include_directories(utils SYSTEM PUBLIC ${MSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(utils PRIVATE ${MSIM_INCLUDE_DIRS})

target_link_libraries(utils PRIVATE fmt::fmt)
