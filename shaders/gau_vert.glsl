#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 g_pos;
layout(location = 2) in vec4 g_rot;
layout(location = 3) in vec3 g_scale;
layout(location = 4) in vec3 g_dc_color;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out vec3 color;

void main()
{
    color = g_dc_color;

    vec3 pos = position + g_rot.yzw;
    pos = pos * g_scale;
    pos = pos + g_pos;
    gl_Position = projection_matrix * view_matrix * vec4(pos, 1.0);
}
