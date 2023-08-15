
#include "MapGlObject.hpp"
#include "sim/Entity.hpp"
#include "sim/Map.hpp"
#include "sim/Simulator.hpp"
#include "sim/Config.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace ui::widgets::opengl {
void MapGlObject::init_internal() {
    assert(simulator);
    const std::shared_ptr<sim::Map> map = simulator->get_map();
    if (!map) {
        return;
    }

    // Vertex data:
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(sim::RoadPiece) * map->roadPieces.size()), static_cast<void*>(map->roadPieces.data()), GL_DYNAMIC_DRAW);
    GLERR;

    std::filesystem::path programFilePath = sim::Config::working_directory() / "assets/shader/gtk/map.bin";

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
        vertShader = compile_shader("/ui/shader/map/map.vert", GL_VERTEX_SHADER);
        assert(vertShader > 0);
        fragShader = compile_shader("/ui/shader/map/map.frag", GL_FRAGMENT_SHADER);
        assert(fragShader > 0);

        // Prepare program:
        glAttachShader(shaderProg, vertShader);
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
            SPDLOG_ERROR("Linking map shader program failed: {}", log_msg);
            glDeleteProgram(shaderProg);
            shaderProg = 0;
            std::exit(1);
        } else {
            glDetachShader(shaderProg, fragShader);
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

    // Bind attributes:
    glUseProgram(shaderProg);
    GLint colAttrib = glGetAttribLocation(shaderProg, "color");
    GLERR;
    glEnableVertexAttribArray(colAttrib);
    GLERR;
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
    glVertexAttribPointer(colAttrib, 4, GL_FLOAT, GL_FALSE, sizeof(sim::RoadPiece), reinterpret_cast<void*>(2 * sizeof(sim::Vec2)));
    GLERR;

    GLint posAttrib = glGetAttribLocation(shaderProg, "position");
    glEnableVertexAttribArray(posAttrib);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(sim::RoadPiece), nullptr);
    GLERR;

    GLint worldSizeConst = glGetUniformLocation(shaderProg, "worldSize");
    GLERR;
    glUniform2f(worldSizeConst, sim::Config::map_width, sim::Config::map_height);
    GLERR;
}

void MapGlObject::render_internal() {
    assert(simulator);
    const std::shared_ptr<sim::Map> map = simulator->get_map();
    if (!map) {
        return;
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(sim::RoadPiece) * map->roadPieces.size()), static_cast<void*>(map->roadPieces.data()));

    glLineWidth(1);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(map->roadPieces.size()));
}

void MapGlObject::cleanup_internal() {
    glDeleteShader(fragShader);
    glDeleteShader(fragShader);
}
}  // namespace ui::widgets::opengl
