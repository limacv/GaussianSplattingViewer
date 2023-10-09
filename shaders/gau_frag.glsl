#version 330 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;  // local coordinate in quad, unit in pixel

out vec4 FragColor;

void main()
{
    FragColor = vec4(color, alpha);
}
