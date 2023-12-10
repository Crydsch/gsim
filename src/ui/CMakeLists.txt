cmake_minimum_required(VERSION 3.16)

add_subdirectory(widgets)
add_subdirectory(windows)
add_subdirectory(resources)

add_library(ui UiContext.cpp)

target_include_directories(ui SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(ui PRIVATE ${GSIM_INCLUDE_DIRS})

# Propagate the variables upwards
set(RESOURCE_DEF ${RESOURCE_DEF} PARENT_SCOPE)
set(RESOURCE_DIR ${RESOURCE_DIR} PARENT_SCOPE)
set(RESOURCE_DEPS ${RESOURCE_DEPS} PARENT_SCOPE)

target_link_libraries(ui
    PUBLIC PkgConfig::GTKMM ${ADWAITA_LIBRARIES} ui_windows
    PRIVATE logger
)
