// water.frag
#version 330 core
out vec4 FragColor;

in vec3 worldPos;
in vec3 normal;
in vec2 uv;
in vec4 reflectionPos;
in vec4 refractionPos;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

uniform sampler2D foamMap;
uniform samplerCube skybox;

uniform sampler2D reflectionTex;
uniform sampler2D refractionTex;
uniform sampler2D refractionDepthTex;
uniform sampler2D foamTexture;

uniform float nearPlane;
uniform float farPlane;
uniform vec2 screenSize;
uniform float time;
uniform float waterHeight;

uniform mat4 inverseProjection;
uniform mat4 inverseView;

void main() {
    vec3 dx = dFdx(worldPos);
    vec3 dy = dFdy(worldPos);
    vec3 normal = normalize(cross(dy, dx));
    float refl_strength = .03;
    float refr_strength = .03;
    vec2 reflectionCoord = reflectionPos.xy * 0.5 + 0.5;
    reflectionCoord += refl_strength * normal.xz / reflectionPos.z;
    vec2 refractionCoord  = refractionPos.xy * 0.5 + 0.5;
    refractionCoord += refr_strength * normal.xz / refractionPos.z;

    vec3 lightDir = normalize(lightPos - worldPos);
    vec3 viewDir = normalize(viewPos - worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * lightColor;

    vec3 R = reflect(-viewDir, normal);
    vec3 skyColor = texture(skybox, R).rgb;

    float fresnel = clamp(1.0 - dot(viewDir, normal), 0.0, 1.0);
    vec4 reflection = texture(reflectionTex, clamp(reflectionCoord, vec2(0.001), vec2(0.999))).rgba;
    vec4 refraction = texture(refractionTex, clamp(refractionCoord, vec2(0.001), vec2(0.999))).rgba;
    reflection = vec4(mix(skyColor + specular, reflection.rgb, reflection.a), 1.);
    //FragColor = vec4(mix(refraction.rgb, reflection.rgb, fresnel), 1.);
    //return;

    // calculate water depth
    vec2 screenUV = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(refractionDepthTex, screenUV).r;
    float z = sceneDepth * 2.0 - 1.0;
    float viewDepth = (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
    float waterDepth = viewDepth - length(viewPos - worldPos);
    float depthFactor = clamp(waterDepth / 2.0, 0.0, 1.0);

    //restore tarrain height
    vec2 ndc = screenUV * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, z, 1.0);
    vec4 viewPos4 = inverseProjection * clipPos;
    viewPos4 /= viewPos4.w;
    vec4 worldPos4 = inverseView * viewPos4;
    float terrainHeight = worldPos4.y;

    float verticalDepth = waterHeight - terrainHeight;
    float foamEdge = smoothstep(0.1, 0.0, verticalDepth);
    foamEdge = pow(foamEdge, 0.5);
    float staticFoam = texture(foamMap, uv).r;
    float foamNoise = texture(foamTexture, uv * 4.0 + vec2(time * 0.05)).r;
    //float foamWave = sin(10.0 * uv.x + time * 2.0) * 0.5 + 0.5;
    float dynamicFoam = foamEdge * foamNoise;

    staticFoam = 0.;
    float totalFoam = clamp(staticFoam + dynamicFoam, 0.0, 1.0);
    totalFoam = pow(totalFoam, 0.4);

    vec3 depthTint = vec3(0.3, 0.8, 0.9);
    vec3 refractionColor = refraction.rgb * (1.0 - depthFactor) + depthTint * depthFactor;

    vec3 finalColor = mix(refractionColor, reflection.rgb, fresnel);
    finalColor = mix(finalColor, vec3(1.0), totalFoam);
    FragColor = vec4(finalColor, 1.0);
}

