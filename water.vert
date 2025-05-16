// water.vert
#version 330 core
layout (location = 0) in vec3 aPos;

uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform sampler2D heightMap;
uniform sampler2D normalMap;

out vec3 worldPos;
out vec3 normal;

void main() {
    vec3 pos = aPos;
    vec2 uv = (pos.xz + 1.0) / 2.0;
    pos.y = texture(heightMap, uv).r;
    normal = normalize(texture(normalMap, uv).rgb * 2.0 - 1.0);

    worldPos = vec3(model * vec4(pos, 1.0));
    gl_Position = projection * view * vec4(worldPos, 1.0);
}
