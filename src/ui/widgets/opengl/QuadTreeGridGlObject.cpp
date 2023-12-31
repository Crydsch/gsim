
#include "QuadTreeGridGlObject.hpp"
#include "sim/Entity.hpp"
#include "sim/GpuQuadTree.hpp"
#include "sim/Config.hpp"
#include <cassert>
#include <epoxy/gl_generated.h>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace ui::widgets::opengl {
void QuadTreeGridGlObject::set_quad_tree_nodes(const sim::gpu_quad_tree::Node* nodes) {
    // Transform to points:
    vertices.clear();
    GLsizei newVerticesCount = 0;
    add_node_rec(nodes, nodes[0], newVerticesCount);
    bool sameSize = newVerticesCount == verticesCount;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLERR;

    // Only change the buffer size in case the number of elements actually changed:
    if (sameSize) {
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(sim::Vec2) * vertices.size()), static_cast<void*>(vertices.data()));
    } else {
        verticesCount = newVerticesCount;
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(sim::Vec2) * vertices.size()), static_cast<void*>(vertices.data()), GL_DYNAMIC_DRAW);
    }
    GLERR;
}

void QuadTreeGridGlObject::add_node_rec(const sim::gpu_quad_tree::Node* nodes, const sim::gpu_quad_tree::Node& node, GLsizei& newVerticesCount) {
    vertices.push_back({node.offsetX, node.offsetY});  // Top Left
    vertices.push_back({node.offsetX + node.width, node.offsetY});  // Top Right

    vertices.push_back({node.offsetX + node.width, node.offsetY});  // Top Right
    vertices.push_back({node.offsetX + node.width, node.offsetY + node.height});  // Bottom Right

    vertices.push_back({node.offsetX + node.width, node.offsetY + node.height});  // Bottom Right
    vertices.push_back({node.offsetX, node.offsetY + node.height});  // Bottom Left

    vertices.push_back({node.offsetX, node.offsetY + node.height});  // Bottom Left
    vertices.push_back({node.offsetX, node.offsetY});  // Top Left

    newVerticesCount += 8;

    // If inner node
    if (node.entityCount > 0 && node.first >= sim::Config::num_entities) {
        add_node_rec(nodes, nodes[node.nextTL], newVerticesCount);
        add_node_rec(nodes, nodes[node.nextTL + 1], newVerticesCount);
        add_node_rec(nodes, nodes[node.nextTL + 2], newVerticesCount);
        add_node_rec(nodes, nodes[node.nextTL + 3], newVerticesCount);
    }
}

void QuadTreeGridGlObject::init_internal() {
    // Vertex data:
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    GLERR;

    std::filesystem::path programFilePath = sim::Config::working_directory() / "assets/shader/gtk/quadTreeGrid.bin";

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
        vertShader = compile_shader("/ui/shader/quadTreeGrid/quadTreeGrid.vert", GL_VERTEX_SHADER);
        assert(vertShader > 0);
        fragShader = compile_shader("/ui/shader/quadTreeGrid/quadTreeGrid.frag", GL_FRAGMENT_SHADER);
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
            SPDLOG_ERROR("Linking quad tree grid shader program failed: {}", log_msg);
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

    GLint posAttrib = glGetAttribLocation(shaderProg, "position");
    glEnableVertexAttribArray(posAttrib);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(sim::Vec2), nullptr);
    GLERR;

    worldSizeConst = glGetUniformLocation(shaderProg, "worldSize");
    glUniform2f(worldSizeConst, sim::Config::map_height, sim::Config::map_height);
    GLERR;
}

void QuadTreeGridGlObject::render_internal() {
    glLineWidth(10);
    glDrawArrays(GL_LINES, 0, verticesCount);
}

void QuadTreeGridGlObject::cleanup_internal() {
    glDeleteShader(fragShader);
    glDeleteShader(fragShader);
}
}  // namespace ui::widgets::opengl