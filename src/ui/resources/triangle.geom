#version 330 core

uniform vec2 worldSize;
uniform vec2 rectSize;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
out vec3 fColor;

in vec2 vPosition[];
out vec2 fPosition;

void build_rect(vec4 position, vec2 size) {
    size /= 2;
    gl_Position = position + vec4(- size.x, -size.y, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4(size.x, -size.y, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4(-size.x, size.y, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4(size.x, size.y, 0.0, 0.0);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    fColor = vColor[0];
    fPosition = vPosition[0];
    vec2 size = 2 * (rectSize / worldSize);
    build_rect(gl_in[0].gl_Position, size);
}