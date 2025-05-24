#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in ivec4 aJoints;
layout(location = 3) in vec4 aWeights;

uniform mat4 uMat[9];
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;
void main() {
    mat4 skinMat = aWeights.x * uMat[aJoints.x] +
        aWeights.y * uMat[aJoints.y] +
        aWeights.z * uMat[aJoints.z] +
        aWeights.w * uMat[aJoints.w];
        
    vec4 skPos = skinMat * vec4(aPos,1.0);
    FragPos = vec3(skPos);
    mat3 normMat = mat3(transpose(inverse(skinMat)));
    Normal = normalize(normMat * aNormal);
    gl_Position = projection * view * skPos;
}