cmake_minimum_required(VERSION 3.16)

add_library(ui_widgets_opengl_fb
    AbstractGlFrameBuffer.cpp
    EntitiesFrameBuffer.cpp
    MapFrameBuffer.cpp
    QuadTreeGridFrameBuffer.cpp)

target_include_directories(ui_widgets_opengl_fb SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(ui_widgets_opengl_fb PRIVATE ${GSIM_INCLUDE_DIRS})

target_link_libraries(ui_widgets_opengl_fb PRIVATE logger PkgConfig::EPOXY PkgConfig::GTKMM ui_widgets_opengl_utils)
