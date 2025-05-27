// water.vert
#version 330 core
layout (location = 0) in vec3 aPos;

uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 rainDrops[32];
uniform int rainCount;

uniform sampler2D heightMap;
uniform sampler2D normalMap;

float baseHeight(vec2 uv) {
    return texture(heightMap, uv).r;
}

float rippleContribution(vec2 p, vec2 dropP, float dt) {
    float dist = length(p - dropP);
    float speed = 0.2;                   // 波速
    float wavelength = 0.2;              // 波長
    float k = 2.0 * 3.14159 / wavelength;
    float omega = k * speed;

    if (dt <= 0.0 || dt * speed < dist) {
        return 0.0;
    }

    float damp = exp(-2.0 * dt);
    return 0.1 * damp * cos(k*dist - omega*dt) / (1.0 + dist*10.0);
}

out vec3 worldPos;
out vec3 normal;

void main() {
    vec3 pos = aPos;
    vec2 uv = (pos.xz + 1.0) / 2.0;
    float h = baseHeight(uv);

    for (int i = 0; i < rainCount; ++i) {
        h += rippleContribution(uv, vec2(rainDrops[i].x, rainDrops[i].y), time - rainDrops[i].z);
    }

    pos.y = h;

    normal = normalize(texture(normalMap, uv).rgb * 2.0 - 1.0);

    worldPos = vec3(model * vec4(pos, 1.0));
    gl_Position = projection * view * vec4(worldPos, 1.0);
}
