// water.frag
#version 330 core
out vec4 FragColor;
in vec3 fragPos;

void main() {
    float height = fragPos.y;
    vec3 waterColor = mix(vec3(0.0, 0.3, 0.6), vec3(0.2, 0.6, 0.9), 0.5 + 0.5 * height);

    // 加白沫效果：高於一定高度
    if (height > 0.07)
        waterColor += vec3(0.8);

    FragColor = vec4(waterColor, 1.0);
}

