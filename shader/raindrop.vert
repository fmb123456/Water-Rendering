#version 330 core
layout (location = 0) in vec2 quadVertex; // [-0.5 ~ 0.5]
layout (location = 1) in vec3 instancePos;

uniform mat4 view;
uniform mat4 projection;

out vec2 uv;

void main() {
    vec3 camRight = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 camUp    = vec3(view[0][1], view[1][1], view[2][1]);

    float size = 0.05;
    vec3 pos = instancePos + quadVertex.x * camRight * size + quadVertex.y * camUp * size;

    gl_Position = projection * view * vec4(pos, 1.0);
    uv = quadVertex + vec2(0.5);
}


