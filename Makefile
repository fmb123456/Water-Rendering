all:water_rendering.cpp
	g++ water_rendering.cpp src/glad.c -Iinclude -lglfw -ldl -lGL -o water_rendering

clean:
	rm main

