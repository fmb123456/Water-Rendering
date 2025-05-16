// water_rendering.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "shader.h"

// --- Function Prototypes ---
void initOpenGL();
void renderGround();
void renderWaterSurface(float time);
void renderFoamTexture();
void renderExtraEffects(float time);
void renderScene(float time);
void processInput(GLFWwindow* window);

// Global variables
unsigned int groundVAO = 0, waterVAO = 0;
unsigned int groundVBO = 0, waterVBO = 0;
GLFWwindow* window;
std::vector<glm::vec3> waterVertices;
Shader *waterShader, *groundShader;

const int GRID_SIZE = 100;

// --- Main ---
int main() {
    initOpenGL();
    groundShader = new Shader("ground.vert", "ground.frag");
    waterShader = new Shader("water.vert", "water.frag");

    float time = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        time = (float)glfwGetTime();
        processInput(window);

        glClearColor(0.1f, 0.3f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderGround();
        renderWaterSurface(time);
        renderFoamTexture();
        renderExtraEffects(time);
        renderScene(time);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

// --- OpenGL Initialization ---
void initOpenGL() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(800, 600, "Water Rendering", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    // Ground quad setup
    float groundVertices[] = {
        -1.0f, 0.0f, -1.0f,
         1.0f, 0.0f, -1.0f,
         1.0f, 0.0f,  1.0f,
        -1.0f, 0.0f,  1.0f
    };
    glGenVertexArrays(1, &groundVAO);
    glGenBuffers(1, &groundVBO);
    glBindVertexArray(groundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, groundVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(groundVertices), groundVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Create water grid
    for (int z = 0; z < GRID_SIZE - 1; ++z) {
        for (int x = 0; x < GRID_SIZE - 1; ++x) {
            float x0 = -1.0f + 2.0f * x / GRID_SIZE;
            float z0 = -1.0f + 2.0f * z / GRID_SIZE;
            float x1 = -1.0f + 2.0f * (x + 1) / GRID_SIZE;
            float z1 = -1.0f + 2.0f * (z + 1) / GRID_SIZE;

            waterVertices.push_back(glm::vec3(x0, 0.0f, z0));
            waterVertices.push_back(glm::vec3(x1, 0.0f, z0));
            waterVertices.push_back(glm::vec3(x1, 0.0f, z1));

            waterVertices.push_back(glm::vec3(x0, 0.0f, z0));
            waterVertices.push_back(glm::vec3(x1, 0.0f, z1));
            waterVertices.push_back(glm::vec3(x0, 0.0f, z1));
        }
    }

    glGenVertexArrays(1, &waterVAO);
    glGenBuffers(1, &waterVBO);
    glBindVertexArray(waterVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferData(GL_ARRAY_BUFFER, waterVertices.size() * sizeof(glm::vec3), &waterVertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
}

// --- Input Processing ---
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// --- 1. Render Ground ---
void renderGround() {
    groundShader->use();
    glBindVertexArray(groundVAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

// --- 2. Render Water Surface ---
void renderWaterSurface(float time) {
    std::vector<glm::vec3> animatedVertices = waterVertices;
    for (auto& v : animatedVertices) {
        v.y = 0.05f * sinf(10.0f * v.x + time) + 0.05f * cosf(10.0f * v.z + time);
    }

    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, animatedVertices.size() * sizeof(glm::vec3), &animatedVertices[0]);

    waterShader->use();
    waterShader->setFloat("time", time);
    glBindVertexArray(waterVAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawArrays(GL_TRIANGLES, 0, animatedVertices.size());
}

// --- 3. Render Foam Texture ---
void renderFoamTexture() {
    // 可在 water.frag 裡加上基於高度的白沫色彩
}

// --- 4. Render Extra Effects ---
void renderExtraEffects(float time) {
    // 可加入 particle 或 alpha map 效果
}

// --- 5. Render Full Scene ---
void renderScene(float time) {
    // 可在 water.frag 中加入 reflection/refraction 與 Fresnel
}

