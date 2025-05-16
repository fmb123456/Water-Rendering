CXX = g++
CC  = gcc
CXXFLAGS = -Iinclude -Iexternal/glad -Iexternal/glm -Iexternal/glfw/include
LDFLAGS = -ldl -lGL -lX11 -lpthread -lXrandr -lXi
BUILD_DIR = build
TARGET = water_rendering

SRC = \
    water_rendering.cpp \
    src/glad.c

OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC)))

GLFW_LIB = external/glfw/build/src/libglfw3.a

SHADER = water.frag water.vert ground.frag ground.vert skybox.frag skybox.vert

all: $(TARGET)

$(TARGET): $(OBJ) $(GLFW_LIB) $(SHADER)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(GLFW_LIB) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

