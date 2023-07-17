#include "UiContext.hpp"
#include "sim/Config.hpp"
#include <adwaita.h>
#include <gdkmm/display.h>
#include <gtkmm/icontheme.h>
#include <gtkmm/settings.h>

namespace ui {
int UiContext::run() {
    // Initialize Adwaita
    adw_init();

    // Create the main GTK application
    app = Gtk::Application::create("de.msim");

    // Add icon paths
    Gtk::IconTheme::get_for_display(Gdk::Display::get_default())->add_resource_path("/ui/icons/scalable/action");

    app->signal_startup().connect([&] {
        add_main_window();
    });

    // The app will return once execution finished
    return app->run(0, nullptr);
}

void UiContext::add_main_window() {
    if (!mainWindow) {
        mainWindow = std::make_unique<windows::MainWindow>();
    }
    app->add_window(*mainWindow);
    mainWindow->set_visible(true);
}

}  // namespace ui
