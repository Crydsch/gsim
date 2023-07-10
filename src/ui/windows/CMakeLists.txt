cmake_minimum_required(VERSION 3.16)

add_library(ui_windows MainWindow.cpp)

target_include_directories(ui_windows SYSTEM PUBLIC ${MSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(ui_windows PRIVATE ${MSIM_INCLUDE_DIRS})

target_link_libraries(ui_windows PRIVATE PkgConfig::GTKMM logger ui_widgets)
