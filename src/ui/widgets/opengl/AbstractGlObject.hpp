#pragma once

#include "logger/Logger.hpp"
#include "sim/Simulator.hpp"
#include <memory>
#include <string>
#include <epoxy/gl.h>

// Error codes: https://www.khronos.org/opengl/wiki/OpenGL_Error
#define GLERR                                             \
    {                                                     \
        GLuint glErr;                                     \
        while ((glErr = glGetError()) != GL_NO_ERROR) {   \
            SPDLOG_ERROR("glGetError() = 0x{:X}", glErr); \
        }                                                 \
    }

namespace ui::widgets::opengl {
class AbstractGlObject {
 protected:
    GLuint vao{0};
    GLuint vbo{0};

    GLuint shaderProg{0};

    std::shared_ptr<sim::Simulator> simulator{nullptr};

 public:
    AbstractGlObject();
    AbstractGlObject(AbstractGlObject& other) = delete;
    AbstractGlObject(AbstractGlObject&& old) = delete;

    virtual ~AbstractGlObject() = default;

    AbstractGlObject& operator=(AbstractGlObject& other) = delete;
    AbstractGlObject& operator=(AbstractGlObject&& old) = delete;

 protected:
    virtual void init_internal() = 0;
    virtual void render_internal() = 0;
    virtual void cleanup_internal() = 0;

    static GLuint compile_shader(const std::string& resourcePath, GLenum type);

 public:
    void init();
    void render();
    void cleanup();
};
}  // namespace ui::widgets::opengl
