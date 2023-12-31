#pragma once

#include "SimulationOverlayWidget.hpp"
#include "SimulationWidget.hpp"
#include "sim/Simulator.hpp"
#include "ui/widgets/SimulationOverlayWidget.hpp"
#include <memory>
#include <gtkmm/box.h>
#include <gtkmm/switch.h>
#include <gtkmm/togglebutton.h>

namespace ui::widgets {
class SimulationSettingsBarWidget : public Gtk::Box {
 private:
    SimulationWidget* simWidget{nullptr};
    SimulationOverlayWidget* simOverlayWidget{nullptr};

    Gtk::Box mainBox;
    Gtk::Box zoomBox;
    Gtk::Box miscBox;

    Gtk::ToggleButton simulateTBtn;
    Gtk::ToggleButton renderTBtn;
    Gtk::ToggleButton debugOverlayTBtn;

    Gtk::Button zoomInBtn;
    Gtk::Button zoomOutBtn;
    Gtk::Button zoomResetBtn;
    Gtk::Button zoomFitBtn;

    Gtk::ToggleButton blurTBtn;
    Gtk::ToggleButton quadTreeGridTBtn;

    std::shared_ptr<sim::Simulator> simulator{nullptr};

 public:
    SimulationSettingsBarWidget(SimulationWidget* simWidget, SimulationOverlayWidget* simOverlayWidget);

 private:
    void prep_widget();

    //-----------------------------Events:-----------------------------
    void on_simulate_toggled();
    void on_render_toggled();
    void on_debug_overlay_toggled();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_zoom_reset_clicked();
    void on_zoom_fit_clicked();
    void on_blur_toggled();
    void on_quad_tree_grid_toggled();
};
}  // namespace ui::widgets
