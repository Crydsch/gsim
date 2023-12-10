cmake_minimum_required(VERSION 3.16)

add_library(ui_widgets_opengl 
    AbstractGlObject.cpp
    EntityGlObject.cpp
    MapGlObject.cpp
    ScreenSquareGlObject.cpp
    BlurGlObject.cpp
    QuadTreeGridGlObject.cpp)

target_include_directories(ui_widgets_opengl SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(ui_widgets_opengl PRIVATE ${GSIM_INCLUDE_DIRS})

target_link_libraries(ui_widgets_opengl PRIVATE logger PkgConfig::EPOXY PkgConfig::GTKMM ui_widgets_opengl_utils)

add_subdirectory(utils)
add_subdirectory(fb)
