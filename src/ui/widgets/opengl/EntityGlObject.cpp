
#include "EntityGlObject.hpp"
#include "sim/Entity.hpp"
#include "sim/Config.hpp"
#include <cassert>
#include <epoxy/gl_generated.h>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace ui::widgets::opengl {

void EntityGlObject::set_entities(std::vector<sim::Entity>& entities) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    entityCount = static_cast<GLsizei>(entities.size());
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(sim::Entity)) * entityCount, static_cast<void*>(entities.data()));
}

void EntityGlObject::init_internal() {
    // Vertex data
    assert(simulator);
    size_t entities_epoch{42}; // Force initial update
    std::vector<sim::Entity> entities;
    [[maybe_unused]] bool updated = simulator->get_entities(entities, entities_epoch);
    assert(updated);
    entityCount = static_cast<GLsizei>(entities.size());
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(sim::Entity) * entities.size()), static_cast<void*>(entities.data()), GL_DYNAMIC_DRAW);

    std::filesystem::path programFilePath = sim::Config::working_directory() / "assets/shader/gtk/entity.bin";

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
        // Compile shader
        vertShader = compile_shader("/ui/shader/entity/entity.vert", GL_VERTEX_SHADER);
        assert(vertShader > 0);
        geomShader = compile_shader("/ui/shader/entity/entity.geom", GL_GEOMETRY_SHADER);
        assert(geomShader > 0);
        fragShader = compile_shader("/ui/shader/entity/entity.frag", GL_FRAGMENT_SHADER);
        assert(fragShader > 0);

        // Prepare program
        glAttachShader(shaderProg, vertShader);
        glAttachShader(shaderProg, geomShader);
        glAttachShader(shaderProg, fragShader);
        glBindFragDataLocation(shaderProg, 0, "outColor");
        glLinkProgram(shaderProg);
        GLERR;

        // Check for errors during linking
        GLint status = GL_FALSE;
        glGetProgramiv(shaderProg, GL_LINK_STATUS, &status);
        if (status == GL_FALSE) {
            int log_len = 0;
            glGetProgramiv(shaderProg, GL_INFO_LOG_LENGTH, &log_len);

            std::string log_msg;
            log_msg.resize(log_len);
            glGetProgramInfoLog(shaderProg, log_len, nullptr, static_cast<GLchar*>(log_msg.data()));
            SPDLOG_ERROR("Linking entity shader program failed: {}", log_msg);
            glDeleteProgram(shaderProg);
            shaderProg = 0;
            std::exit(1);
        } else {
            glDetachShader(shaderProg, fragShader);
            glDetachShader(shaderProg, geomShader);
            glDetachShader(shaderProg, vertShader);
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

    // Bind attributes
    glUseProgram(shaderProg);
    GLint colAttrib = glGetAttribLocation(shaderProg, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 4, GL_FLOAT, GL_FALSE, sizeof(sim::Entity), nullptr);

    GLint posAttrib = glGetAttribLocation(shaderProg, "position");
    glEnableVertexAttribArray(posAttrib);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(sim::Entity), reinterpret_cast<void*>(sizeof(sim::Rgba)));

    worldSizeConst = glGetUniformLocation(shaderProg, "worldSize");
    glUniform2f(worldSizeConst, sim::Config::map_width, sim::Config::map_height);

    rectSizeConst = glGetUniformLocation(shaderProg, "rectSize");
    glUniform2f(rectSizeConst, 10, 10);
    GLERR;
}

void EntityGlObject::render_internal() {
    glDrawArrays(GL_POINTS, 0, entityCount);
}

void EntityGlObject::cleanup_internal() {
    glDeleteShader(fragShader);
    glDeleteShader(geomShader);
    glDeleteShader(fragShader);
}

}  // namespace ui::widgets::opengl
