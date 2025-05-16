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

    worldPos = vec3(model * vec4(pos, 1.0));
    normal = normalize(vec3(0.0, 1.0, 0.0)); // Simplified normal

    gl_Position = projection * view * vec4(worldPos, 1.0);
}
