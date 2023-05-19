
#include "BlurGlObject.hpp"
#include "sim/Entity.hpp"
#include <array>
#include <cassert>
#include <epoxy/gl_generated.h>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace ui::widgets::opengl {
void BlurGlObject::bind_texture(GLuint inputTexture) {
    this->inputTexture = inputTexture;
}

void BlurGlObject::set_texture_size(GLsizei sizeX, GLsizei sizeY) {
    inputTextureSizeX = sizeX;
    inputTextureSizeY = sizeY;
}

void BlurGlObject::init_internal() {
    std::filesystem::path programFilePath = "/home/crydsch/msim/assets/shader/gtk/blur.bin"; // TODO use config

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
        vertShader = compile_shader("/ui/shader/blur/blur.vert", GL_VERTEX_SHADER);
        assert(vertShader > 0);
        geomShader = compile_shader("/ui/shader/blur/blur.geom", GL_GEOMETRY_SHADER);
        assert(geomShader > 0);
        fragShader = compile_shader("/ui/shader/blur/blur.frag", GL_FRAGMENT_SHADER);
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
            SPDLOG_ERROR("Linking blur shader program failed: {}", log_msg);
            glDeleteProgram(shaderProg);
            shaderProg = 0;
            std::exit(1);
        } else {
            glDetachShader(shaderProg, vertShader); // TODO delete shader?
            glDetachShader(shaderProg, geomShader);
            glDetachShader(shaderProg, fragShader);
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
    glUniform1i(glGetUniformLocation(shaderProg, "inputTexture"), 0);

    // Bind uniform offsets buffer
    glGenBuffers(1, &offsetsUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, offsetsUBO);
    GLERR;
    assert(inputTextureSizeX > 0);
    assert(inputTextureSizeY > 0);
    float stepX = 1 / static_cast<float>(inputTextureSizeX);
    float stepY = 1 / static_cast<float>(inputTextureSizeY);
    std::array<sim::Vec2, 9> data{{{-stepX, -stepY}, {0, -stepY}, {stepX, -stepY}, {-stepX, 0}, {0, 0}, {stepX, 0}, {-stepX, stepY}, {0, stepY}, {stepX, stepY}}};
    glBufferData(GL_UNIFORM_BUFFER, sizeof(sim::Vec2) * data.size(), data.data(), GL_STATIC_DRAW);
    GLERR;
    GLuint offsetsIndex = glGetUniformBlockIndex(shaderProg, "blurArrayBlock");
    GLERR;
    glUniformBlockBinding(shaderProg, offsetsIndex, 0);
    GLERR;
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, offsetsUBO);
    GLERR;
}

void BlurGlObject::render_internal() {
    glBindTexture(GL_TEXTURE_2D, inputTexture);
    glDrawArrays(GL_POINTS, 0, 1);
    GLERR;
}

void BlurGlObject::cleanup_internal() {
    glDeleteShader(vertShader);
    glDeleteShader(geomShader);
    glDeleteShader(fragShader);
}
}  // namespace ui::widgets::opengl