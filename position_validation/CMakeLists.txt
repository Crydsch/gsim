cmake_minimum_required(VERSION 3.20)

set(PROJECT_DESCRIPTION "Position validation")
project("positionValidation" 
            VERSION 1.0.0
            DESCRIPTION "${PROJECT_DESCRIPTION}")
set(VERSION_NAME "dev")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(position_validation_option OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    option(${OPTION_NAME} ${OPTION_TEXT} ${OPTION_DEFAULT})
    if(DEFINED ENV{${OPTION_NAME}})
        # Allow setting the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()
    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()
    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

message(STATUS "Position validation options")
message(STATUS "=======================================================")
position_validation_option(POSITION_VALIDATION_STATIC_ANALYZE "Set to ON to enable the GCC 10 static analysis. If enabled, POSITION_VALIDATION_ENABLE_LINTING has to be disabled." ON)
position_validation_option(POSITION_VALIDATION_ENABLE_LINTING "Set to ON to enable clang linting. If enabled, POSITION_VALIDATION_STATIC_ANALYZE has to be disabled." OFF)
message(STATUS "=======================================================")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(sanitizer)
include(gcc_analyze)

if(POSITION_VALIDATION_ENABLE_LINTING)
    message(STATUS "Enabling linting")
    include(clang-tidy)
else()
    message(STATUS "Linting is disabled")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)