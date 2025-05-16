// water_rendering.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <fftw3.h>
#include "shader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// --- Function Prototypes ---
void initOpenGL();
void renderGround();
void renderWaterSurface(float time);
void renderFoamTexture();
void renderExtraEffects(float time);
void renderScene(float time);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

unsigned int loadCubemap(std::vector<std::string> faces);
void renderSkybox(const glm::mat4& view, const glm::mat4& projection);

// Global variables
unsigned int groundVAO = 0, waterVAO = 0;
unsigned int groundVBO = 0, waterVBO = 0;
GLFWwindow* window;
std::vector<glm::vec3> waterVertices;
Shader *waterShader, *groundShader;

unsigned int skyboxVAO, skyboxVBO;
unsigned int cubemapTexture;
Shader *skyboxShader;

unsigned int heightTex, normalTex;

const int GRID_SIZE = 300;

// Camera
glm::vec3 cameraPos = glm::vec3(0.0f, 0.5f, 2.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float lastX = 400.0f, lastY = 300.0f;
bool firstMouse = true;
float yaw = -90.0f, pitch = 0.0f, fov = 45.0f;

// Wave Simulate
struct Wave {
    using Complex = std::complex<float>;

    const int N = 100;
    const float L = 2.0f;
    const float A = 0.05f;
    const float G = 9.81f;
    const glm::vec2 wind = glm::vec2(2.0f, 2.0f);

    std::vector<Complex> h0;
    std::vector<Complex> hkt;
    std::vector<float> heightMap;
    std::vector<glm::vec3> normalMap;

    Wave(): h0(N * N), hkt(N * N), heightMap(N * N), normalMap(N * N) {
        initializeH0();
    }

    float gaussianRandom() {
        static std::default_random_engine eng;
        static std::normal_distribution<float> dist(0.0f, 1.0f);
        return dist(eng);
    }

    float phillips(glm::vec2 k) {
        float kLen = glm::length(k);
        if (kLen < 1e-6f) return 0.0f;

        float kLen2 = kLen * kLen;
        float kLen4 = kLen2 * kLen2;
        glm::vec2 kNorm = glm::normalize(k);
        float kDotW = glm::dot(kNorm, glm::normalize(wind));
        float Lw = glm::length(wind);
        float L = (Lw * Lw) / G;
        float L2 = L * L;
        float damping = 0.001f;
        float l2 = (L * damping) * (L * damping);

        return A * exp(-1.0f / (kLen2 * L2)) / kLen4 * kDotW * kDotW * exp(-kLen2 * l2);
    }

    glm::vec2 getWaveVector(int n, int m) {
        float kx = 2.0f * glm::pi<float>() * (m - N / 2) / L;
        float ky = 2.0f * glm::pi<float>() * (n - N / 2) / L;
        return glm::vec2(kx, ky);
    }

    void initializeH0() {
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < N; ++m) {
                glm::vec2 k = getWaveVector(n, m);
                float P = phillips(k);
                float r1 = gaussianRandom();
                float r2 = gaussianRandom();
                h0[n * N + m] = Complex(r1, r2) * sqrt(P / 2.0f);
            }
        }
    }

    void updateFrequencyDomain(float t) {
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < N; ++m) {
                glm::vec2 k = getWaveVector(n, m);
                int idx = n * N + m;
                Complex h0k = h0[idx];
                int mi = (N - m) % N, ni = (N - n) % N;
                Complex h0minusk = std::conj(h0[ni * N + mi]);
                float kLen = glm::length(k);
                float omega = sqrt(G * kLen);
                Complex e = exp(Complex(0, omega * t));
                hkt[idx] = h0k * e + h0minusk * std::conj(e);
            }
        }
    }

    void computeHeightMap() {
        fftwf_complex* in = reinterpret_cast<fftwf_complex*>(hkt.data());
        fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N * N);
        fftwf_plan plan = fftwf_plan_dft_2d(N, N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        float Min = 0.0;
        for (int i = 0; i < N * N; ++i) {
            heightMap[i] = out[i][0];
            heightMap[i] = abs(heightMap[i]);
        }
        fftwf_destroy_plan(plan);
        fftwf_free(out);
    }

    void computeNormalMap(float spacing) {
        for (int z = 0; z < N; ++z) {
            for (int x = 0; x < N; ++x) {
                int idx = z * N + x;
                float hL = heightMap[z * N + (x > 0 ? x - 1 : x)];
                float hR = heightMap[z * N + (x < N - 1 ? x + 1 : x)];
                float hD = heightMap[(z > 0 ? z - 1 : z) * N + x];
                float hU = heightMap[(z < N - 1 ? z + 1 : z) * N + x];
                float dx = (hR - hL) / (2.0f * spacing);
                float dz = (hU - hD) / (2.0f * spacing);
                normalMap[idx] = glm::normalize(glm::vec3(-dx, 1.0f, -dz));
            }
        }
    }
    void uploadHeightToGPU(GLuint texID) {
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RED, GL_FLOAT, heightMap.data());
    }

    void uploadNormalsToGPU(GLuint texID) {
        std::vector<glm::vec3> rgbData(N * N);
        for (int i = 0; i < N * N; ++i)
            rgbData[i] = normalMap[i] * 0.5f + 0.5f;

        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RGB, GL_FLOAT, rgbData.data());
    }
} wave;

// --- Main ---
int main() {
    initOpenGL();
    groundShader = new Shader("ground.vert", "ground.frag");
    waterShader = new Shader("water.vert", "water.frag");

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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
        
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
        renderSkybox(view, projection);

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
        -1.0f, -0.2f, -1.0f,
         1.0f, -0.2f, -1.0f,
         1.0f, -0.2f,  1.0f,
        -1.0f, -0.2f,  1.0f
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
    
    // Skybox cube vertices ([-1, 1])
    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    // generate skybox VAO/VBO
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    // load cubemap texture
    std::vector<std::string> faces{
        "skybox/right.jpg",
        "skybox/left.jpg",
        "skybox/top.jpg",
        "skybox/bottom.jpg",
        "skybox/front.jpg",
        "skybox/back.jpg"
    };
    cubemapTexture = loadCubemap(faces);
    
    // load skybox shader
    skyboxShader = new Shader("skybox.vert", "skybox.frag");


    // init wave shape texture
    glGenTextures(1, &heightTex);
    glBindTexture(GL_TEXTURE_2D, heightTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, wave.N, wave.N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenTextures(1, &normalTex);
    glBindTexture(GL_TEXTURE_2D, normalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, wave.N, wave.N, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}

// --- Input Processing ---
void processInput(GLFWwindow* window) {
    const float cameraSpeed = 2.5f * 0.016f;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    yaw += xoffset;
    pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// --- 0. Render Skybox ---
void renderSkybox(const glm::mat4& view, const glm::mat4& projection) {
    glDepthFunc(GL_LEQUAL);  // change it to LEQUAL and make sure skybox passes the depth test
    
    skyboxShader->use();
    // remove the panning component of the view and keep only the rotation, so that the skybox always wraps around the camera
    glm::mat4 viewNoTranslate = glm::mat4(glm::mat3(view));
    skyboxShader->setMat4("view", viewNoTranslate);
    skyboxShader->setMat4("projection", projection);
    
    glBindVertexArray(skyboxVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    skyboxShader->setFloat("skybox", 0);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glDepthFunc(GL_LESS);  // restore deep test mode

    waterShader->use();
    waterShader->setFloat("skybox", 0);
}

unsigned int loadCubemap(std::vector<std::string> faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format = nrChannels == 4 ? GL_RGBA : GL_RGB;
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        } else {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}


// --- 1. Render Ground ---
void renderGround() {
    groundShader->use();
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
    groundShader->setMat4("model", model);
    groundShader->setMat4("view", view);
    groundShader->setMat4("projection", projection);
    glBindVertexArray(groundVAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

// --- 2. Render Water Surface ---
void renderWaterSurface(float time) {
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, waterVertices.size() * sizeof(glm::vec3), &waterVertices[0]);

    wave.updateFrequencyDomain(time / 10.0);
    wave.computeHeightMap();
    wave.computeNormalMap(wave.L / wave.N);

    waterShader->use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);

    glActiveTexture(GL_TEXTURE1);
    wave.uploadHeightToGPU(heightTex);
    glUniform1i(glGetUniformLocation(waterShader->ID, "heightMap"), 1);

    glActiveTexture(GL_TEXTURE2);
    wave.uploadNormalsToGPU(normalTex);
    glUniform1i(glGetUniformLocation(waterShader->ID, "normalMap"), 2);

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
    waterShader->setMat4("model", model);
    waterShader->setMat4("view", view);
    waterShader->setMat4("projection", projection);
    waterShader->setFloat("time", time);
    waterShader->setVec3("lightPos", glm::vec3(2.0f, 2.0f, 2.0f));
    waterShader->setVec3("viewPos", cameraPos);
    waterShader->setVec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    glBindVertexArray(waterVAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawArrays(GL_TRIANGLES, 0, waterVertices.size());
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

