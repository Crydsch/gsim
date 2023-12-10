cmake_minimum_required(VERSION 3.20)

# Fix install RPATH for Linux:
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

set(MAIN_EXECUTABLE ${PROJECT_NAME})

add_subdirectory(logger)
add_subdirectory(utils)
add_subdirectory(sim)

if(MOVEMENT_SIMULATOR_ENABLE_GUI)
    add_subdirectory(ui)
endif()

# We use a timestamp file to watch the resource files for changes
#  On change we delete the cached shaders to force re-compilation
add_custom_command(
    OUTPUT clean_cached_shaders.stamp
    COMMAND ${CMAKE_COMMAND} -E touch clean_cached_shaders.stamp
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_SOURCE_DIR}/assets/shader/gtk
    DEPENDS ${RESOURCE_DEPS}
    COMMENT "Cleaning cached gtk shaders"
)

add_custom_target(clean_cached_shaders
    DEPENDS clean_cached_shaders.stamp
)

if(MOVEMENT_SIMULATOR_ENABLE_GUI)
    glib_add_resource_file(TARGET ui_resources
                        RESOURCEDEF ${RESOURCE_DEF}
                        WORKING_DIR ${RESOURCE_DIR}
                        OUTFILE ui_resources.c
                        DEPENDS ${RESOURCE_DEPS})
    set(UI_RESOURCES ${CMAKE_CURRENT_BINARY_DIR}/ui_resources.c)
endif()

add_executable(${MAIN_EXECUTABLE}
    main.cpp
    ${UI_RESOURCES}
)
add_dependencies(${MAIN_EXECUTABLE} clean_cached_shaders)

target_include_directories(${MAIN_EXECUTABLE} SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(${MAIN_EXECUTABLE} PRIVATE ${GSIM_INCLUDE_DIRS})

target_link_libraries(${MAIN_EXECUTABLE} PRIVATE logger sim)
if(MOVEMENT_SIMULATOR_ENABLE_GUI)
target_link_libraries(${MAIN_EXECUTABLE} PRIVATE ui PkgConfig::GTKMM)
endif()
set_property(SOURCE main.cpp PROPERTY COMPILE_DEFINITIONS MOVEMENT_SIMULATOR_VERSION="${PROJECT_VERSION}" MOVEMENT_SIMULATOR_VERSION_NAME="${VERSION_NAME}")

install(TARGETS ${MAIN_EXECUTABLE} RUNTIME DESTINATION)

# Copy final binary to root directory
# add_custom_command(
#         TARGET ${MAIN_EXECUTABLE} POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy
#                 ${CMAKE_CURRENT_BINARY_DIR}/gsim
#                 ${CMAKE_SOURCE_DIR}/gsim)

if(GSIM_CONNECTIVITY_DETECTION STREQUAL "1")
    add_custom_command(
        TARGET ${MAIN_EXECUTABLE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/gsim
                ${CMAKE_SOURCE_DIR}/gsim_cpu_std)

elseif(GSIM_CONNECTIVITY_DETECTION STREQUAL "2")
    add_custom_command(
        TARGET ${MAIN_EXECUTABLE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/gsim
                ${CMAKE_SOURCE_DIR}/gsim_cpu_emil)

elseif(GSIM_CONNECTIVITY_DETECTION STREQUAL "3")
    add_custom_command(
        TARGET ${MAIN_EXECUTABLE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/gsim
                ${CMAKE_SOURCE_DIR}/gsim_cpu)

elseif(GSIM_CONNECTIVITY_DETECTION STREQUAL "4")
    add_custom_command(
        TARGET ${MAIN_EXECUTABLE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/gsim
                ${CMAKE_SOURCE_DIR}/gsim_gpu)

endif()