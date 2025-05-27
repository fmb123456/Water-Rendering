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
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

// --- Function Prototypes ---
void initOpenGL();
void renderGround();
void renderWaterSurface(float time);
void renderFoamTexture();
void renderExtraEffects(float time);
void renderScene(float time);
void renderPool();
void renderBird();
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
unsigned int loadTexture(std::string filePath);

unsigned int loadCubemap(std::vector<std::string> faces);
void renderSkybox();

// Global variables
unsigned int groundVAO = 0, waterVAO = 0;
unsigned int groundVBO = 0, waterVBO = 0;
GLFWwindow* window;
std::vector<glm::vec3> waterVertices;
std::vector<glm::vec3> rainDrops;
const int rainBound = 32, rainDropFreq = 10;
Shader *waterShader, *groundShader;

unsigned int skyboxVAO, skyboxVBO;
unsigned int cubemapTexture;
Shader *skyboxShader;

typedef std::vector<float> VecF;
struct AnimSampler {
    VecF times;              // keyframe times
    std::vector<VecF> values;// outputs per keyframe
    std::string path;        // translation/rotation/scale/weights
    int targetIndex;         // node or primitive index
};

struct object {
    unsigned int vao, vbo, ebo, indexCount;
    std::vector<AnimSampler> animSamplers;
    tinygltf::Model model;
} pool, bird;

void loadObject(object &obj, std::string filePath);

Shader *poolShader, *birdShader;
unsigned int stoneTextureID, woodTextureID;

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

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    float time = 0.0f;
    std::default_random_engine eng;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int dropCnt = (int)(dist(eng) * rainDropFreq);
    while (!glfwWindowShouldClose(window)) {
        time = (float)glfwGetTime();
        processInput(window);

        glClearColor(0.1f, 0.3f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderSkybox();
        renderGround();
        renderWaterSurface(time);
        if (dropCnt == 0) {
            dropCnt = (int)(dist(eng) * rainDropFreq);
            rainDrops.push_back(glm::vec3(dist(eng), dist(eng), time));
            if (rainDrops.size() > rainBound) {
                rainDrops.erase(rainDrops.begin());
            }
        } else {
            dropCnt--;
        }
        renderFoamTexture();
        renderExtraEffects(time);
        renderScene(time);
        renderPool();
        renderBird();
        
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
    groundShader = new Shader("ground.vert", "ground.frag");

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
    waterShader = new Shader("water.vert", "water.frag");

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

    // pool
    loadObject(pool, "model/terrain.glb");
    poolShader = new Shader("pool.vert", "pool.frag");

    // bird
    loadObject(bird, "model/bird.glb");
    birdShader = new Shader("bird.vert", "bird.frag");

    // texture
    stoneTextureID = loadTexture("texture/stone.png");
    woodTextureID = loadTexture("texture/wood.jpg");
}

unsigned int loadTexture(std::string filePath) {
    int width, height, nrChannels;
    stbi_uc* data = stbi_load(filePath.c_str(),
                              &width, &height, &nrChannels,
                              0);
    if (!data) {
        std::cerr << "Failed to load texture: " << filePath << "\n";
        exit(0);
    }

    unsigned int textureID;

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format = GL_RGB;
    if (nrChannels == 1)      format = GL_RED;
    else if (nrChannels == 3) format = GL_RGB;
    else if (nrChannels == 4) format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D,
                 0,               // mipmap level
                 format,          // internal format
                 width, height,
                 0,               // border
                 format,          // source format
                 GL_UNSIGNED_BYTE,
                 data);

    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}

void loadObject(object &obj, std::string filePath) {
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret = loader.LoadBinaryFromFile(&obj.model, &err, &warn, filePath);
    auto prim = obj.model.meshes[0].primitives[0];

    if (!obj.model.animations.empty()) {
        auto anim = obj.model.animations[0];

        for (size_t i = 0; i < anim.samplers.size(); ++i) {
            const auto& sam = anim.samplers[i];
            AnimSampler as;
            // read times
            auto& accIn = obj.model.accessors[sam.input];
            auto& viewIn = obj.model.bufferViews[accIn.bufferView];
            auto& bufIn = obj.model.buffers[viewIn.buffer];
            const float* tptr = reinterpret_cast<const float*>(bufIn.data.data() + viewIn.byteOffset + accIn.byteOffset);
            as.times.assign(tptr, tptr + accIn.count);
            // read values
            auto& accOut = obj.model.accessors[sam.output];
            auto& viewOut = obj.model.bufferViews[accOut.bufferView];
            auto& bufOut = obj.model.buffers[viewOut.buffer];
            const float* vptr = reinterpret_cast<const float*>(bufOut.data.data() + viewOut.byteOffset + accOut.byteOffset);
            size_t comp = accOut.type; // SCALAR=1, VEC3=3, VEC4=4
            as.values.resize(accOut.count);
            for (size_t f = 0; f < accOut.count; ++f) {
                as.values[f].assign(vptr + f*comp, vptr + (f+1)*comp);
            }
            // find channel target
            for (auto& channel : anim.channels) {
                if (channel.sampler == i) {
                    as.targetIndex = channel.target_node;
                    as.path = channel.target_path;
                }
            }
            obj.animSamplers.push_back(as);
        }
    }

    const auto& posAccessor = obj.model.accessors[prim.attributes.at("POSITION")];
    const auto& posView = obj.model.bufferViews[posAccessor.bufferView];
    const auto& posBuffer = obj.model.buffers[posView.buffer];
    const float* positions = reinterpret_cast<const float*>(&posBuffer.data[posView.byteOffset + posAccessor.byteOffset]);

    // Normals (optional)
    const float* normals = nullptr;
    const unsigned char* joints = nullptr;
    const float* weights = nullptr;
    const float* texcoord = nullptr;
    if (prim.attributes.count("NORMAL")) {
        auto &nAccessor = obj.model.accessors[prim.attributes.at("NORMAL")];
        auto &nView = obj.model.bufferViews[nAccessor.bufferView];
        auto &nBuffer = obj.model.buffers[nView.buffer];
        normals = reinterpret_cast<const float*>(&nBuffer.data[nView.byteOffset + nAccessor.byteOffset]);
    }
    if (prim.attributes.count("JOINTS_0")) {
        auto &nAccessor = obj.model.accessors[prim.attributes.at("JOINTS_0")];
        auto &nView = obj.model.bufferViews[nAccessor.bufferView];
        auto &nBuffer = obj.model.buffers[nView.buffer];
        joints = reinterpret_cast<const unsigned char*>(&nBuffer.data[nView.byteOffset + nAccessor.byteOffset]);
    }
    if (prim.attributes.count("WEIGHTS_0")) {
        auto &nAccessor = obj.model.accessors[prim.attributes.at("WEIGHTS_0")];
        auto &nView = obj.model.bufferViews[nAccessor.bufferView];
        auto &nBuffer = obj.model.buffers[nView.buffer];
        weights = reinterpret_cast<const float*>(&nBuffer.data[nView.byteOffset + nAccessor.byteOffset]);
    }
    if (prim.attributes.count("TEXCOORD_0")) {
        auto &nAccessor = obj.model.accessors[prim.attributes.at("TEXCOORD_0")];
        auto &nView = obj.model.bufferViews[nAccessor.bufferView];
        auto &nBuffer = obj.model.buffers[nView.buffer];
        texcoord = reinterpret_cast<const float*>(&nBuffer.data[nView.byteOffset + nAccessor.byteOffset]);
    }

    // Indices
    const auto& idxAccessor = obj.model.accessors[prim.indices];
    const auto& idxView = obj.model.bufferViews[idxAccessor.bufferView];
    const auto& idxBuffer = obj.model.buffers[idxView.buffer];
    const unsigned short* indices = reinterpret_cast<const unsigned short*>(&idxBuffer.data[idxView.byteOffset + idxAccessor.byteOffset]);
    obj.indexCount = idxAccessor.count;

    // Upload to GPU
    glGenVertexArrays(1, &obj.vao);
    glBindVertexArray(obj.vao);
    glGenBuffers(1, &obj.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, obj.vbo);
    glBufferData(GL_ARRAY_BUFFER, posAccessor.count * 3 * sizeof(float), positions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    if (normals) {
        GLuint nbo;
        glGenBuffers(1, &nbo);
        glBindBuffer(GL_ARRAY_BUFFER, nbo);
        glBufferData(GL_ARRAY_BUFFER, posAccessor.count * 3 * sizeof(float), normals, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);
    }
    if (joints) {
        GLuint nbo;
        glGenBuffers(1, &nbo);
        glBindBuffer(GL_ARRAY_BUFFER, nbo);
        glBufferData(GL_ARRAY_BUFFER, posAccessor.count * 4 * sizeof(unsigned char), joints, GL_STATIC_DRAW);
        glVertexAttribIPointer(2, 4, GL_UNSIGNED_BYTE, 0, nullptr);
        glEnableVertexAttribArray(2);
    }
    if (weights) {
        GLuint nbo;
        glGenBuffers(1, &nbo);
        glBindBuffer(GL_ARRAY_BUFFER, nbo);
        glBufferData(GL_ARRAY_BUFFER, posAccessor.count * 4 * sizeof(float), weights, GL_STATIC_DRAW);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(3);
    }
    if (texcoord) {
        GLuint nbo;
        glGenBuffers(1, &nbo);
        glBindBuffer(GL_ARRAY_BUFFER, nbo);
        glBufferData(GL_ARRAY_BUFFER, posAccessor.count * 2 * sizeof(float), texcoord, GL_STATIC_DRAW);
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(4);
    }
    glGenBuffers(1, &obj.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.indexCount * sizeof(unsigned short), indices, GL_STATIC_DRAW);
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
void renderSkybox() {
    glDepthFunc(GL_LEQUAL);  // change it to LEQUAL and make sure skybox passes the depth test
    
    skyboxShader->use();
    // remove the panning component of the view and keep only the rotation, so that the skybox always wraps around the camera
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 viewNoTranslate = glm::mat4(glm::mat3(view));
    glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
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

void renderPool() {
    glDepthFunc(GL_LEQUAL);  // change it to LEQUAL and make sure skybox passes the depth test
    poolShader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, stoneTextureID);
    {
        GLint loc = glGetUniformLocation(birdShader->ID, "uTexture");
        glUniform1i(loc, 0);
    }

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(0.12));
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
    poolShader->setMat4("model", model);
    poolShader->setMat4("projection", projection);
    poolShader->setMat4("view", view);
    poolShader->setVec3("viewPos", cameraPos);
    glm::mat4 modelMat = glm::mat4(1.0f);

    glBindVertexArray(pool.vao);
    glDrawElements(GL_TRIANGLES, pool.indexCount, GL_UNSIGNED_SHORT, nullptr);

    glDepthFunc(GL_LESS);  // restore deep test mode
}

VecF interpolate(const AnimSampler& as, float t) {
    // boundary
    if (t <= as.times.front()) return as.values.front();
    if (t >= as.times.back())  return as.values.back();
    // find interval
    size_t idx = 0;
    while (idx + 1 < as.times.size() && t > as.times[idx+1]) ++idx;
    float t0 = as.times[idx], t1 = as.times[idx+1];
    float f = (t - t0)/(t1 - t0);
    const auto& v0 = as.values[idx];
    const auto& v1 = as.values[idx+1];
    VecF out(v0.size());
    for (size_t j = 0; j < v0.size(); ++j) out[j] = v0[j]*(1-f) + v1[j]*f;
    return out;
}

void renderBird() {
    glDepthFunc(GL_LEQUAL);  // change it to LEQUAL and make sure skybox passes the depth test
    birdShader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, woodTextureID);
    {
        GLint loc = glGetUniformLocation(birdShader->ID, "uTexture");
        glUniform1i(loc, 0);
    }

    auto animSamplers = bird.animSamplers;

    float t = fmod(glfwGetTime(), animSamplers.front().times.back() - 0.1) + 0.05;

    for (auto& as : animSamplers) {
        VecF val = interpolate(as,t);

        auto& node = bird.model.nodes[as.targetIndex];
        if (as.path == "translation") {
            node.translation = { val[0], val[1], val[2] };
        } else if (as.path == "rotation") {
            node.rotation = { val[0], val[1], val[2], val[3] };
        } else if (as.path == "scale") {
            node.scale = { val[0], val[1], val[2] };
        }
    }

    std::vector<glm::mat4> nodeMatrices(bird.model.nodes.size(), glm::mat4(1.0f));

    std::function<void(int, const glm::mat4&)> computeNode = [&](int idx, const glm::mat4& parentMatrix) {
        const auto& node = bird.model.nodes[idx];
        glm::mat4 local = glm::mat4(1.0f);
        if (idx < 9) {
            local = glm::translate(local, glm::vec3(
                node.translation[0], node.translation[1], node.translation[2]
            ));
            local *= glm::mat4_cast(glm::quat(
                node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]
            ));
            local = glm::scale(local, glm::vec3(
                node.scale[0], node.scale[1], node.scale[2]
            ));
        }
        glm::mat4 world = parentMatrix * local;
        nodeMatrices[idx] = world;
        for (int child : node.children) {
            computeNode(child, world);
        }
    };

    for (int rootIndex : bird.model.scenes[0].nodes) {
        computeNode(rootIndex, glm::mat4(1.0f));
    }

    const auto& skin = bird.model.skins[0];
    std::vector<glm::mat4> invBindMats(skin.joints.size());
    {
        const auto& ibAccessor = bird.model.accessors[skin.inverseBindMatrices];
        const auto& ibView     = bird.model.bufferViews[ibAccessor.bufferView];
        const auto& ibBuffer   = bird.model.buffers[ibView.buffer];
        const float* ibData = reinterpret_cast<const float*>(
              &ibBuffer.data[ibView.byteOffset + ibAccessor.byteOffset]
        );
        for (size_t i = 0; i < skin.joints.size(); ++i) {
            invBindMats[i] = glm::make_mat4(&ibData[i * 16]);
        }
    }

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(0.12f));
    model = glm::translate(model, glm::vec3(0.0f, 10.0f, 0.0f));

    float step = fmod(glfwGetTime(), 30.0);
    if (step <= 10) {
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, step * 3 - 15.0f));
    } else if (step <= 15) {
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 15.0f));
        model = glm::rotate(model, glm::radians((step - 10.0f) * 180.0f / 5), glm::vec3(0.0f, 1.0f, 0.0f));
    } else if (step <= 25) {
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 60.f - step * 3));
        model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    } else {
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, -15.0f));
        model = glm::rotate(model, glm::radians((30.0f - step) * 180.0f / 5), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    std::vector<glm::mat4> jointMats(skin.joints.size());
    for (size_t i = 0; i < skin.joints.size(); ++i) {
        int nodeIndex = skin.joints[i];
        glm::mat4 world = nodeMatrices[nodeIndex];
        jointMats[i] = model * world * invBindMats[i];
    }

    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
    birdShader->setMat4("model", model);
    birdShader->setMat4("projection", projection);
    birdShader->setMat4("view", view);
    birdShader->setVec3("viewPos", cameraPos);
    {
        GLint loc = glGetUniformLocation(birdShader->ID, "uMat");
        glUniformMatrix4fv(loc, 9, GL_FALSE, glm::value_ptr(jointMats[0]));
    }

    glBindVertexArray(bird.vao);
    glDrawElements(GL_TRIANGLES, bird.indexCount, GL_UNSIGNED_SHORT, nullptr);

    glDepthFunc(GL_LESS);  // restore deep test mode
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
    {
        GLint loc = glGetUniformLocation(waterShader->ID, "rainDrops");
        glUniform3fv(loc, (int)rainDrops.size(), glm::value_ptr(rainDrops[0]));
    }
    {
        GLint loc = glGetUniformLocation(waterShader->ID, "rainCount");
        glUniform1i(loc, (int)rainDrops.size());
    }
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

