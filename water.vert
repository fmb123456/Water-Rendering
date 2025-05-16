// water.vert
#version 330 core
layout (location = 0) in vec3 aPos;
uniform float time;

out vec3 fragPos;

void main() {
    vec3 pos = aPos;
    pos.y += 0.05 * sin(10.0 * pos.x + time) + 0.05 * cos(10.0 * pos.z + time);
    fragPos = pos;
    gl_Position = vec4(pos, 1.0);
}

