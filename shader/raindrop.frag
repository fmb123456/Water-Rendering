#version 330 core
in vec2 uv;
out vec4 FragColor;

uniform sampler2D rainTexture;

void main() {
    vec4 tex = texture(rainTexture, uv);
    if (tex.a < 0.1) discard;
    FragColor = tex;
}

