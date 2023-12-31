#pragma once

#include <memory>
#include <gtkmm.h>

#include "windows/MainWindow.hpp"

namespace ui
{
class UiContext
{
 private:
    std::unique_ptr<windows::MainWindow> mainWindow{nullptr};

 public:
    Glib::RefPtr<Gtk::Application> app{nullptr};

    int run();

    void add_main_window();
};
}  // namespace ui
