#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in int ind;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform mat4 model0;
uniform mat4 model1;
uniform mat4 model2;
uniform mat4 model3;
uniform mat4 model4;
uniform mat4 model5;
uniform mat4 model6;
uniform mat4 model7;
uniform mat4 model8;

out vec3 Normal;
out vec3 FragPos;
out float tmp;
void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
    tmp = ind;
}