#include "SimulationWidget.hpp"
#include "logger/Logger.hpp"
#include "sim/Entity.hpp"
#include "sim/Map.hpp"
#include "sim/Simulator.hpp"
#include "sim/Config.hpp"
#include "spdlog/fmt/bundled/core.h"
#include "spdlog/spdlog.h"
#include "ui/widgets/opengl/fb/MapFrameBuffer.hpp"
#include "ui/widgets/opengl/fb/QuadTreeGridFrameBuffer.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <bits/chrono.h>
#include <epoxy/gl.h>
#include <epoxy/gl_generated.h>
#include <fmt/core.h>
#include <glibconfig.h>
#include <gtkmm/gestureclick.h>
#include <gtkmm/gesturezoom.h>

namespace ui::widgets {
SimulationWidget::SimulationWidget() : simulator(sim::Simulator::get_instance()),
                                       mapFrameBuffer(sim::Config::max_render_resolution_x, sim::Config::max_render_resolution_y),
                                       entitiesFrameBuffer(sim::Config::max_render_resolution_x, sim::Config::max_render_resolution_y),
                                       quadTreeGridFrameBuffer(sim::Config::max_render_resolution_x, sim::Config::max_render_resolution_y) {
    prep_widget();
}

void SimulationWidget::set_zoom_factor(float zoomFactor) {
    assert(zoomFactor > 0);

    this->zoomFactor = zoomFactor;
    float widthF = static_cast<float>(sim::Config::max_render_resolution_x) * this->zoomFactor;
    int width = static_cast<int>(widthF);
    float heightF = static_cast<float>(sim::Config::max_render_resolution_y) * this->zoomFactor;
    int height = static_cast<int>(heightF);
    glArea.set_size_request(width, height);
    glArea.queue_draw();
}

void SimulationWidget::prep_widget() {
    set_expand();

    glArea.signal_render().connect(sigc::mem_fun(*this, &SimulationWidget::on_render_handler), true);
    glArea.signal_realize().connect(sigc::mem_fun(*this, &SimulationWidget::on_realized));
    glArea.signal_unrealize().connect(sigc::mem_fun(*this, &SimulationWidget::on_unrealized));
    glArea.add_tick_callback(sigc::mem_fun(*this, &SimulationWidget::on_tick));
    Glib::RefPtr<Gtk::GestureClick> clickGesture = Gtk::GestureClick::create();
    clickGesture->set_button(GDK_BUTTON_PRIMARY);
    clickGesture->signal_pressed().connect(sigc::mem_fun(*this, &SimulationWidget::on_glArea_clicked));
    glArea.add_controller(clickGesture);
    glArea.set_auto_render();
    glArea.set_size_request(sim::Config::max_render_resolution_x, sim::Config::max_render_resolution_y);
    set_child(glArea);
    screenSquareObj.set_glArea(&glArea);
}

const utils::TickRate& SimulationWidget::get_fps() const {
    return fps;
}

const utils::TickDurationHistory& SimulationWidget::get_fps_history() const {
    return fpsHistory;
}

float SimulationWidget::get_zoom_factor() const {
    return zoomFactor;
}

void SimulationWidget::set_blur(bool blur) {
    this->blur = blur;
}

void SimulationWidget::set_quad_tree_grid_visibility(bool quadTreeGridVisible) {
    this->quadTreeGridVisible = quadTreeGridVisible;
    screenSquareObj.set_quad_tree_grid_visibility(quadTreeGridVisible);
}

//-----------------------------Events:-----------------------------
bool SimulationWidget::on_render_handler(const Glib::RefPtr<Gdk::GLContext>& /*ctx*/) {
    assert(simulator);

    std::chrono::high_resolution_clock::time_point frameStart = std::chrono::high_resolution_clock::now();

    try {
        glArea.throw_if_error();

        // Update the data on the GPU:
        bool entitiesChanged = false;
        bool quadTreeNodesChanged = false;

        if (enableUiUpdates) {
            entitiesChanged = simulator->get_entities(&entities, entitiesEpoch);
            quadTreeNodesChanged = simulator->get_quad_tree_nodes(&quadTreeNodes, quadTreeNodesEpoch);
        }
        
        // Get default frame buffer since in GTK it is not always 0:
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &defaultFb);

        // Backup the old viewport:
        std::array<int, 4> viewPort{};
        glGetIntegerv(GL_VIEWPORT, viewPort.data());

        // Draw:
        glDisable(GL_DEPTH_TEST);

        // 1.0 Draw map if needed:
        if (!mapRendered) {
            mapRendered = true;

            mapFrameBuffer.bind();
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            GLERR;

            mapObj.render();
        }

        // 2.0 Draw entities to buffer:
        if (entitiesChanged) {
            entitiesFrameBuffer.bind();
            // 2.1 Blur old entities:
            if (blur) {
                blurObject.render();
            } else {
                glClearColor(0, 0, 0, 0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                GLERR;
            }

            // 2.2 Draw entities:
            entityObj.set_entities(this->entities);
            entityObj.render();
        }

        // 3.0 Draw quad tree to buffer:
        if (quadTreeNodesChanged && quadTreeGridVisible) {
            quadTreeGridFrameBuffer.bind();
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            GLERR;

            quadTreeGridGlObj.set_quad_tree_nodes(quadTreeNodes);
            GLERR;
            quadTreeGridGlObj.render();
            GLERR;
        }

        // 4.0 Draw to screen:
        glBindFramebuffer(GL_FRAMEBUFFER, defaultFb);
        GLERR;

        // Fix view port so it does not only show values in range [-1,0]:
        glViewport(viewPort[0], viewPort[1], viewPort[2], viewPort[3]);

        // 4.1 Clear the old screen:
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        GLERR;

        // 4.2 Draw texture from frame buffer:
        screenSquareObj.render();
    } catch (const Gdk::GLError& gle) {
        SPDLOG_ERROR("An error occurred in the render callback of the GLArea: {} - {} - {}", gle.domain(), gle.code(), gle.what());
    }

    std::chrono::high_resolution_clock::time_point frameEnd = std::chrono::high_resolution_clock::now();
    fpsHistory.add_time(frameEnd - frameStart);

    // FPS counter:
    fps.tick();
    return false;
}

bool SimulationWidget::on_tick(const Glib::RefPtr<Gdk::FrameClock>& /*frameClock*/) {
    assert(simulator);

    if (this->enableUiUpdates) {
        glArea.queue_draw();
    }
    return true;
}

void SimulationWidget::on_realized() {
    glArea.make_current();
    try {
        glArea.throw_if_error();

        mapFrameBuffer.init();
        entitiesFrameBuffer.init();
        quadTreeGridFrameBuffer.init();

        // Get default frame buffer since in GTK it is not always 0:
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &defaultFb);
        glBindFramebuffer(GL_FRAMEBUFFER, defaultFb);

        mapObj.init();
        blurObject.set_texture_size(entitiesFrameBuffer.get_texture_size_x(), entitiesFrameBuffer.get_texture_size_y());
        blurObject.init();
        blurObject.bind_texture(entitiesFrameBuffer.get_texture());
        entityObj.init();
        quadTreeGridGlObj.init();
        screenSquareObj.bind_texture(mapFrameBuffer.get_texture(), entitiesFrameBuffer.get_texture(), quadTreeGridFrameBuffer.get_texture());
        screenSquareObj.init();
    } catch (const Gdk::GLError& gle) {
        SPDLOG_ERROR("An error occurred making the context current during realize: {} - {} - {}", gle.domain(), gle.code(), gle.what());
    }
}

void SimulationWidget::on_unrealized() {
    glArea.make_current();
    try {
        glArea.throw_if_error();

        mapObj.cleanup();
        blurObject.cleanup();
        entityObj.cleanup();
        quadTreeGridGlObj.cleanup();
        screenSquareObj.cleanup();

        quadTreeGridFrameBuffer.cleanup();
        entitiesFrameBuffer.cleanup();
        mapFrameBuffer.cleanup();
    } catch (const Gdk::GLError& gle) {
        SPDLOG_ERROR("An error occurred deleting the context current during unrealize: {} - {} - {}", gle.domain(), gle.code(), gle.what());
    }
}

void SimulationWidget::on_glArea_clicked(int /*nPress*/, double x, double y) {
    assert(simulator);
    const std::shared_ptr<sim::Map> map = simulator->get_map();

    // Invert since coordinates are inverted on the map:
    // x = glArea.get_width() - x;
    y = glArea.get_height() - y;

    // Scale up to map size:
    x *= sim::Config::map_width / sim::Config::max_render_resolution_x;
    y *= sim::Config::map_height / sim::Config::max_render_resolution_y;
    sim::Vec2 pos{static_cast<float>(x), static_cast<float>(y)};

    if (!map || map->roads.empty()) {
        return;
    }

    // Get closest road:
    size_t roadIndex = 0;
    double shortestDist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < map->roads.size(); i++) {
        // Start:
        double newDist = map->roads[i].start.pos.dist(pos);
        assert(newDist >= 0);
        if (newDist < shortestDist) {
            shortestDist = newDist;
            roadIndex = i;
        }

        // End:
        newDist = map->roads[i].end.pos.dist(pos);
        assert(newDist >= 0);
        if (newDist < shortestDist) {
            shortestDist = newDist;
            roadIndex = i;
        }
    }

    map->select_road(roadIndex);
    // Ensure we rerender the map once the road selection changed:
    mapRendered = false;

    float x1 = map->roads[roadIndex].start.pos.x;
    float x2 = map->roads[roadIndex].end.pos.x;
    float y1 = map->roads[roadIndex].start.pos.y;
    float y2 = map->roads[roadIndex].end.pos.y;
    SPDLOG_DEBUG("Road ({}) selected between position ({}|{}) and ({}|{}) with distance of {} meters.", roadIndex, x1, y1, x2, y2, shortestDist);
}
}  // namespace ui::widgets
