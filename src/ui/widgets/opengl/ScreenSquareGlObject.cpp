
#include "ScreenSquareGlObject.hpp"
#include "sim/Entity.hpp"
#include "sim/Config.hpp"
#include <cassert>
#include <epoxy/gl_generated.h>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace ui::widgets::opengl {
void ScreenSquareGlObject::set_glArea(Gtk::GLArea* glArea) {
    this->glArea = glArea;
}

void ScreenSquareGlObject::bind_texture(GLuint mapFrameBufferTexture, GLuint entitiesFrameBufferTexture, GLuint quadTreeGridFrameBufferTexture) {
    frameBufferTextures = {mapFrameBufferTexture, entitiesFrameBufferTexture, quadTreeGridFrameBufferTexture};
}

void ScreenSquareGlObject::init_internal() {
    std::filesystem::path programFilePath = sim::Config::working_directory() / "assets/shader/gtk/screen_square.bin";

    shaderProg = glCreateProgram();
    bool shaderProgReady = false;

    // Load program from file if possible
    if (std::filesystem::exists(programFilePath) && std::filesystem::is_regular_file(programFilePath)) 
    {
        // Load shader
        uintmax_t fileSize = std::filesystem::file_size(programFilePath);
        std::vector<uint8_t> programData(fileSize);
        GLenum programBinaryFormat = 0;
        std::ifstream programFile;
        programFile.open(programFilePath, std::ios::binary);
        // read binary format first - then program bytes
        programFile.read((char *)&programBinaryFormat, sizeof(GLenum));
        programFile.read((char *)programData.data(), fileSize - sizeof(GLenum));
        programFile.close();

        glProgramBinary(shaderProg, programBinaryFormat, programData.data(), fileSize - sizeof(GLenum));
        if(glGetError() != GL_NO_ERROR)
        {
            SPDLOG_DEBUG("Loading shader program '{}' failed.", programFilePath.c_str());
        }
        else
        {
            SPDLOG_DEBUG("Loading shader program '{}' was successful.", programFilePath.c_str());
            shaderProgReady = true;
        }
    }

    if (!shaderProgReady) {
        // Compile shader:
        vertShader = compile_shader("/ui/shader/screen_square/screen_square.vert", GL_VERTEX_SHADER);
        assert(vertShader > 0);
        geomShader = compile_shader("/ui/shader/screen_square/screen_square.geom", GL_GEOMETRY_SHADER);
        assert(geomShader > 0);
        fragShader = compile_shader("/ui/shader/screen_square/screen_square.frag", GL_FRAGMENT_SHADER);
        assert(fragShader > 0);

        // Prepare program:
        glAttachShader(shaderProg, vertShader);
        glAttachShader(shaderProg, geomShader);
        glAttachShader(shaderProg, fragShader);
        glBindFragDataLocation(shaderProg, 0, "outColor");
        glLinkProgram(shaderProg);
        GLERR;

        // Check for errors during linking:
        GLint status = GL_FALSE;
        glGetProgramiv(shaderProg, GL_LINK_STATUS, &status);
        if (status == GL_FALSE) {
            int log_len = 0;
            glGetProgramiv(shaderProg, GL_INFO_LOG_LENGTH, &log_len);

            std::string log_msg;
            log_msg.resize(log_len);
            glGetProgramInfoLog(shaderProg, log_len, nullptr, static_cast<GLchar*>(log_msg.data()));
            SPDLOG_ERROR("Linking screen square shader program failed: {}", log_msg);
            glDeleteProgram(shaderProg);
            shaderProg = 0;
            std::exit(1);
        } else {
            glDetachShader(shaderProg, fragShader);
            glDetachShader(shaderProg, vertShader);
            glDetachShader(shaderProg, geomShader);
        }
        GLERR;

        // Save compiled program to file for caching
        GLint programSize = 0;
        glGetProgramiv(shaderProg, GL_PROGRAM_BINARY_LENGTH, &programSize);
        GLERR;
        std::vector<uint8_t> programData(programSize);
        GLsizei actualProgramSize = 0;
        GLenum programBinaryFormat = 0;
        glGetProgramBinary(shaderProg, programSize, &actualProgramSize, &programBinaryFormat, programData.data());
        GLERR;

        std::filesystem::create_directories(programFilePath.parent_path());
        std::ofstream programFile;
        programFile.open(programFilePath, std::ios::binary | std::ios::trunc);
        // write binary format first - then program bytes
        programFile.write((const char *)&programBinaryFormat, sizeof(GLenum));
        programFile.write((const char *)programData.data(), programData.size());
        programFile.close();
    }

    // Bind attributes:
    glUseProgram(shaderProg);
    glUniform1i(glGetUniformLocation(shaderProg, "mapTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProg, "entitiesTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProg, "quadTreeGridTexture"), 2);

    textureSizeConst = glGetUniformLocation(shaderProg, "textureSize");
    glUniform2f(textureSizeConst, sim::Config::max_render_resolution_x, sim::Config::max_render_resolution_y);

    screenSizeConst = glGetUniformLocation(shaderProg, "screenSize");
    assert(glArea);
    glUniform2f(screenSizeConst, static_cast<float>(glArea->get_width()), static_cast<float>(glArea->get_height()));

    quadTreeGridVisibleConst = glGetUniformLocation(shaderProg, "quadTreeGridVisible");
    glUniform1ui(quadTreeGridVisibleConst, 0);
    GLERR;
}

void ScreenSquareGlObject::render_internal() {
    assert(glArea);
    glUniform2f(screenSizeConst, static_cast<float>(glArea->get_width()), static_cast<float>(glArea->get_height()));
    glBindTextures(0, static_cast<GLsizei>(frameBufferTextures.size()), frameBufferTextures.data());
    glDrawArrays(GL_POINTS, 0, 1);
    GLERR;
}

void ScreenSquareGlObject::cleanup_internal() {
    glDeleteShader(fragShader);
    glDeleteShader(geomShader);
    glDeleteShader(fragShader);
}

void ScreenSquareGlObject::set_quad_tree_grid_visibility(bool quadTreeGridVisible) const {
    glUseProgram(shaderProg);
    glUniform1ui(quadTreeGridVisibleConst, quadTreeGridVisible ? 1 : 0);
}
}  // namespace ui::widgets::opengl