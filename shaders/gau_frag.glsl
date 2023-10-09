#version 330 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;  // local coordinate in quad, unit in pixel

out vec4 FragColor;

void main()
{
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (power > 0.f)
        discard;
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;
    FragColor = vec4(color, opacity);
}
