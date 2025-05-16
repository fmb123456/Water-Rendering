// water.vert
#version 330 core
layout (location = 0) in vec3 aPos;

uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 worldPos;
out vec3 normal;

void main() {
    vec3 pos = aPos;
    pos.y += 0.05 * sin(10.0 * pos.x + time) + 0.05 * cos(10.0 * pos.z + time);

    vec3 offsetX = vec3(0.01, 0.0, 0.0);
    vec3 offsetZ = vec3(0.0, 0.0, 0.01);
    float hL = 0.05 * sin(10.0 * (pos.x - offsetX.x) + time) + 0.05 * cos(10.0 * pos.z + time);
    float hR = 0.05 * sin(10.0 * (pos.x + offsetX.x) + time) + 0.05 * cos(10.0 * pos.z + time);
    float hD = 0.05 * sin(10.0 * pos.x + time) + 0.05 * cos(10.0 * (pos.z - offsetZ.z) + time);
    float hU = 0.05 * sin(10.0 * pos.x + time) + 0.05 * cos(10.0 * (pos.z + offsetZ.z) + time);

    vec3 dx = vec3(2.0 * offsetX.x, hR - hL, 0.0);
    vec3 dz = vec3(0.0, hU - hD, 2.0 * offsetZ.z);
    normal = normalize(cross(dz, dx));

    worldPos = vec3(model * vec4(pos, 1.0));
    gl_Position = projection * view * vec4(worldPos, 1.0);
}
