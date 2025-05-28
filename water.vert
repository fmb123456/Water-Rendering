// water.vert
#version 330 core
layout (location = 0) in vec3 aPos;

uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 rainDrops[32];
uniform int rainCount;

uniform float waterHeight;
uniform mat4 reflectionView;
uniform mat4 refractionModel;

uniform sampler2D heightMap;
uniform sampler2D normalMap;

out vec3 worldPos;
out vec3 normal;
out vec2 uv;
out vec2 reflectionCoord;
out vec2 refractionCoord;

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

void main() {
    vec3 pos = aPos;
    uv = (pos.xz + 1.0) / 2.0;
    float h = baseHeight(uv);
    //normal = normalize(texture(normalMap, uv).rgb * 2.0 - 1.0);

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

    vec4 undisplacedPos = vec4(aPos, 1.);
    undisplacedPos.y = waterHeight;
    float refl_strength = .1;
    float refr_strength = .1;
    vec4 screenPos = projection * reflectionView * vec4(vec3(model * undisplacedPos), 1.0);
    screenPos /= screenPos.w;
    reflectionCoord = screenPos.xy * 0.5 + 0.5;
    reflectionCoord += refl_strength * normal.xz / screenPos.z;
    screenPos = projection * view * vec4(vec3(refractionModel * undisplacedPos), 1.0);
    screenPos /= screenPos.w;
    refractionCoord  = screenPos.xy * 0.5 + 0.5;
    refractionCoord += refr_strength * normal.xz / screenPos.z;
}

