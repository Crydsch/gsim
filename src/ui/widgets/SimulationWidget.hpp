#pragma once

#include "sim/Entity.hpp"
#include "sim/Simulator.hpp"
#include "utils/TickDurationHistory.hpp"
#include "utils/TickRate.hpp"
#include <memory>
#include <epoxy/gl.h>
#include <gtkmm.h>
#include <gtkmm/glarea.h>
#include <gtkmm/scrolledwindow.h>

namespace ui::widgets {
class SimulationWidget : public Gtk::ScrolledWindow {
 private:
    std::shared_ptr<sim::Simulator> simulator{nullptr};
    std::shared_ptr<std::vector<sim::Entity>> entities{nullptr};

    utils::TickDurationHistory fpsHistory{};
    utils::TickRate fps{};

    // OpenGL:
    GLuint vbo{0};
    GLuint prog{0};
    GLuint vao{0};
    GLint worldSizeConst{0};
    GLint rectSizeConst{0};
    GLint viewPortConst{0};

    GLuint vertShader{0};
    GLuint geomShader{0};
    GLuint fragShader{0};

    Gtk::GLArea glArea;

 public:
    bool enableUiUpdates{true};

    SimulationWidget();

    [[nodiscard]] const utils::TickRate& get_fps() const;
    [[nodiscard]] const utils::TickDurationHistory& get_fps_history() const;

 private:
    static GLuint compile_shader(const std::string& resourcePath, GLenum type);
    void prep_widget();
    void prepare_shader();
    void prepare_buffers();
    void bind_attributes();

    //-----------------------------Events:-----------------------------
    bool on_render_handler(const Glib::RefPtr<Gdk::GLContext>& ctx);
    bool on_tick(const Glib::RefPtr<Gdk::FrameClock>& frameClock);
    void on_realized();
    void on_unrealized();
    void on_adjustment_changed();
};
}  // namespace ui::widgets
