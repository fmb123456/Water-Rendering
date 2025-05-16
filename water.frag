// water.frag
#version 330 core
out vec4 FragColor;

in vec3 worldPos;
in vec3 normal;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

void main() {
    vec3 lightDir = normalize(lightPos - worldPos);
    vec3 viewDir = normalize(viewPos - worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    vec3 waterColor = vec3(0.0, 0.4, 0.7);
    vec3 ambient = 0.1 * lightColor;
    vec3 diffuse = diff * lightColor;
    vec3 specular = spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * waterColor;

    // white foam effect at higher y positions
    if (worldPos.y > 0.07)
        result += vec3(0.8);

    FragColor = vec4(result, 1.0);
}
