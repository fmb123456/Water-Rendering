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

float sumRipple(vec2 p, float time) {
    float h = 0.0;
    for (int i = 0; i < rainCount; ++i) {
        float dt = time - rainDrops[i].z;
        h += rippleContribution(p, vec2(rainDrops[i].x, rainDrops[i].y), dt);
    }
    return h;
}

out vec3 worldPos;
out vec3 normal;

void main() {
    vec3 pos = aPos;
    vec2 uv = (pos.xz + 1.0) / 2.0;
    float h = baseHeight(uv);

    h += sumRipple(uv, time);
    pos.y = h;

    float eps = 0.00001;
    float hL = baseHeight(uv + vec2(-eps, 0)) +
               sumRipple(uv + vec2(-eps, 0), time);
    float hR = baseHeight(uv + vec2(+eps, 0)) +
               sumRipple(uv + vec2(+eps, 0), time);
    float hD = baseHeight(uv + vec2(0, -eps)) +
               sumRipple(uv + vec2(0, -eps), time);
    float hU = baseHeight(uv + vec2(0, +eps)) +
               sumRipple(uv + vec2(0, +eps), time);
    vec3 dx = vec3(2.0 * eps, hR - hL, 0.0);
    vec3 dz = vec3(0.0, hU - hD, 2.0 * eps);
    normal = normalize(cross(dz, dx));

    worldPos = vec3(model * vec4(pos, 1.0));
    gl_Position = projection * view * vec4(worldPos, 1.0);
}