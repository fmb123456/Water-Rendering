// water.frag
#version 330 core
out vec4 FragColor;

in vec3 worldPos;
in vec3 normal;
in vec2 uv;
in vec2 reflectionCoord;
in vec2 refractionCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

uniform sampler2D foamMap;
uniform samplerCube skybox;

uniform sampler2D reflectionTex;
uniform sampler2D refractionTex;

uniform sampler2D refractionDepthTex;
uniform float nearPlane;
uniform float farPlane;
uniform vec2 screenSize;

void main() {
    vec3 lightDir = normalize(lightPos - worldPos);
    vec3 viewDir = normalize(viewPos - worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * lightColor;

    vec3 R = reflect(-viewDir, normal);
    vec3 skyColor = texture(skybox, R).rgb;
    float foam = texture(foamMap, uv).r;

    float fresnel = clamp(1.0 - dot(viewDir, normal), 0.0, 1.0);
    vec4 reflection = texture(reflectionTex, clamp(reflectionCoord, vec2(0.001), vec2(0.999))).rgba;
    vec4 refraction = texture(refractionTex, clamp(refractionCoord, vec2(0.001), vec2(0.999))).rgba;
    reflection = vec4(mix(skyColor + specular, reflection.rgb, reflection.a), 1.);
    //FragColor = vec4(mix(refraction.rgb, reflection.rgb, fresnel), 1.);
    //return;

    vec2 screenUV = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(refractionDepthTex, screenUV).r;
    float z = sceneDepth * 2.0 - 1.0;
    float viewDepth = (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
    float waterDepth = viewDepth - length(viewPos - worldPos);

    float depthFactor = clamp(waterDepth / 2.0, 0.0, 1.0);
    vec3 refractionColor = refraction.rgb * (1.0 - depthFactor) + vec3(0.0, 0.2, 0.3) * depthFactor;

    vec3 finalColor = mix(refractionColor, reflection.rgb, fresnel);
    FragColor = vec4(finalColor, 1.0);
}
