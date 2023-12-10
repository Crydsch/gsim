cmake_minimum_required(VERSION 3.16)

add_library(ui_widgets
    SimulationWidget.cpp
    SimulationOverlayWidget.cpp
    SimulationSettingsBarWidget.cpp)

target_include_directories(ui_widgets SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(ui_widgets PRIVATE ${GSIM_INCLUDE_DIRS})

target_link_libraries(ui_widgets PRIVATE PkgConfig::GTKMM logger sim PkgConfig::EPOXY fmt::fmt utils ui_widgets_opengl ui_widgets_opengl_fb)

add_subdirectory(opengl)
