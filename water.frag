// water.frag
#version 330 core
out vec4 FragColor;

in vec3 worldPos;
in vec3 normal;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

uniform samplerCube skybox;

void main() {
    vec3 lightDir = normalize(lightPos - worldPos);
    vec3 viewDir = normalize(viewPos - worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4);

    vec3 waterColor = vec3(0.0, 0.4, 0.7);
    vec3 ambient = 0.1 * lightColor;
    vec3 diffuse = diff * lightColor;
    vec3 specular = spec * lightColor;

    vec3 result = (ambient + diffuse) * waterColor + specular;

    vec3 R = reflect(-viewDir, normal);
    vec3 skyColor = texture(skybox, R).rgb;

    result = 0.5 * (result + skyColor);

    FragColor = vec4(result, 1.0);
}
