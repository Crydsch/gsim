cmake_minimum_required(VERSION 3.20)

set(PROJECT_DESCRIPTION "GSIM stands for GPU-based mobility Simulator.")
project("gsim" 
            VERSION 0.2.0
            DESCRIPTION "${PROJECT_DESCRIPTION}"
            HOMEPAGE_URL "https://github.com/crydsch/gsim")
set(VERSION_NAME "dev")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(gsim_option OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
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

macro(gsim_log_level_option OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    set(${OPTION_NAME} ${OPTION_DEFAULT} CACHE STRING ${OPTION_TEXT})
    set_property(CACHE ${OPTION_NAME} PROPERTY STRINGS "Default" "Trace" "Debug" "Info" "Warn" "Error" "Critical" "Off")
    if(DEFINED ENV{${OPTION_NAME}})
        # Allow setting the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()
    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()
    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

message(STATUS "GSIM client options")
message(STATUS "=======================================================")
gsim_option(GSIM_STATIC_ANALYZE "Set to ON to enable GCC static analysis. If enabled, GSIM_ENABLE_LINTING has to be disabled." OFF)
gsim_option(GSIM_ENABLE_LINTING "Set to ON to enable clang linting. If enabled, GSIM_STATIC_ANALYZE has to be disabled." OFF)
gsim_option(GSIM_ENABLE_RENDERDOC_API "Set to ON to enable RenderDoc in application APIs for Kompute to allow inspecting compute shaders." OFF)
gsim_option(GSIM_SHADER_INTO_HEADER "Set to ON to compile shaders into headers. This increased the build time but shaders do not have to be shipped with the binary." OFF)
gsim_option(GSIM_ENABLE_GUI "Set to ON to enable compiling with gtkmm-4.0 and allow running in GUI mode." OFF)
gsim_log_level_option(GSIM_LOG_LEVEL "Sets the log level. 'Default' will set the log level to 'Debug' for debug builds and 'Info' for release builds." "Default")
message(STATUS "=======================================================")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(sanitizer)
include(gcc_analyze)
include(clear_variable)
include(FetchContent)

if(GSIM_ENABLE_LINTING)
    message(STATUS "Enabling linting")
    include(clang-tidy)
else()
    message(STATUS "Linting is disabled")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

if(GSIM_ENABLE_GUI)
    # gtk4 stuff
    include(glib-resource-compiler)
    find_package(PkgConfig REQUIRED)
    # Additional search path since gtkmm-4.0 will usually be installed in /usr
    set(ENV{PKG_CONFIG_PATH} "/usr/lib/pkgconfig")
    pkg_check_modules(GTKMM gtkmm-4.0>=4.4.0 REQUIRED IMPORTED_TARGET)
    pkg_check_modules(EPOXY epoxy REQUIRED IMPORTED_TARGET)

    # Lib adwaita
    pkg_check_modules(ADWAITA REQUIRED libadwaita-1>=1.1.0)
endif()

if(GSIM_ENABLE_RENDERDOC_API)
    # So CMake can find FindRenderDoc.cmake
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

    message("RenderDoc in application API is enabled.")
    find_package(RenderDoc REQUIRED)
    add_compile_definitions(GSIM_ENABLE_RENDERDOC_API)
endif()

# Disable linting for fetch content projects
clear_variable(DESTINATION CMAKE_CXX_CLANG_TIDY BACKUP CMAKE_CXX_CLANG_TIDY_BKP)
clear_variable(DESTINATION CMAKE_CXX_FLAGS BACKUP CMAKE_CXX_FLAGS_BKP)
clear_variable(DESTINATION CMAKE_CXX_FLAGS_DEBUG BACKUP CMAKE_CXX_FLAGS_DEBUG_BKP)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git
                         GIT_TAG 9.1.0
                         SYSTEM)
FetchContent_MakeAvailable(fmt)

set(SPDLOG_INSTALL OFF CACHE INTERNAL "")
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git
                            GIT_TAG v1.12.0
                            SYSTEM)
FetchContent_MakeAvailable(spdlog)

set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON CACHE INTERNAL "") # Version clash since they are not properly supported on my GPU.
set(KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER ON CACHE INTERNAL "")
set(KOMPUTE_OPT_BUILD_SHADERS ON CACHE INTERNAL "")
set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE INTERNAL "")
FetchContent_Declare(kompute GIT_REPOSITORY https://github.com/Crydsch/kompute.git # fork of https://github.com/KomputeProject/kompute + Custom Patches
                             GIT_TAG 62ac892d224132cc96bbb75f7c4a5151731e2a0b
                             SYSTEM)
FetchContent_MakeAvailable(kompute)

FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git
                          GIT_TAG v3.11.2
                          SYSTEM)
FetchContent_MakeAvailable(json)

restore_variable(DESTINATION CMAKE_CXX_CLANG_TIDY BACKUP CMAKE_CXX_CLANG_TIDY_BKP)
restore_variable(DESTINATION CMAKE_CXX_FLAGS BACKUP CMAKE_CXX_FLAGS_BKP)
restore_variable(DESTINATION CMAKE_CXX_FLAGS_DEBUG BACKUP CMAKE_CXX_FLAGS_DEBUG_BKP)

# Set compile time options (preprocessor definitions)
set(GSIM_BENCHMARK "0" CACHE STRING "Set to 0/1 to disable/enable benchmarking support (utils::Timer)")
set(GSIM_STANDALONE "0" CACHE STRING "Set to 0/1 to disable/enable standalone mode (default is accelerator mode)")
set(GSIM_COPY_REGIONS "1" CACHE STRING "Set to 0/1 to disable/enable support for kompute copy regions")
add_compile_definitions("DEBUG=$<CONFIG:Debug>" "BENCHMARK=${GSIM_BENCHMARK}" "STANDALONE_MODE=${GSIM_STANDALONE}" "KOMPUTE_COPY_REGIONS=${GSIM_COPY_REGIONS}")
if(GSIM_ENABLE_GUI)
add_compile_definitions("ENABLE_GUI=1")
else()
add_compile_definitions("ENABLE_GUI=0")
endif()

set(GSIM_INCLUDE_DIRS_SYSTEM ${ADWAITA_INCLUDE_DIRS} ${fmt_SOURCE_DIR}/include ${kompute_SOURCE_DIR}/src/include)
set(GSIM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src)
