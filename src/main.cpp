#include "main.h"
#include "preview.h"
#include <cstring>
#include <random>

#include "image.h"
#include "utilities.h"
#include "pathtrace.h"

int width;
int height;

static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

glm::vec3* devDirectIllum = nullptr;
glm::vec3* devIndirectIllum = nullptr;
GBuffer gBuffer;

glm::vec3* devImage = nullptr;

LeveledEAWFilter EAWFilter;
SpatioTemporalFilter directFilter;
SpatioTemporalFilter indirectFilter;

void initImageBuffer() {
	devDirectIllum = cudaMalloc<glm::vec3>(width * height);
	devIndirectIllum = cudaMalloc<glm::vec3>(width * height);
	gBuffer.create(width, height);
}

void freeImageBuffer() {
	cudaSafeFree(devDirectIllum);
	cudaSafeFree(devIndirectIllum);
	gBuffer.destroy();
}

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}
	scene = new Scene(argv[1]);

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = scene->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	EAWFilter.create(width, height, 5);
	directFilter.create(width, height, 5);
	indirectFilter.create(width, height, 5);

	scene->buildDevData();
	initImageBuffer();
	pathTraceInit(scene);

	// GLFW main loop
	mainLoop();

	scene->clear();
	Resource::clear();
	freeImageBuffer();
	pathTraceFree();
	EAWFilter.destroy();
	directFilter.destroy();
	indirectFilter.destroy();

	return 0;
}

void saveImage() {
	cudaMemcpyDevToHost(scene->state.image.data(), devImage, width * height * sizeof(glm::vec3));

	float samples = iteration;
	// output image file
	Image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 color = renderState->image[index] / samples;
			switch (Settings::toneMapping) {
			case ToneMapping::Filmic:
				color = Math::filmic(color);
				break;
			case ToneMapping::ACES:
				color = Math::ACES(color);
				break;
			case ToneMapping::None:
				break;
			}
			color = Math::correctGamma(color);
			img.setPixel(width - 1 - x, y, color);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << currentTimeString() << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	State::camChanged = true;
	if (State::camChanged) {
		iteration = 0;
		scene->camera.update();
		State::camChanged = false;
	}

	gBuffer.render(scene->devScene, scene->camera);

	uchar4* devPBO = nullptr;
	cudaGLMapBufferObject((void**)&devPBO, pbo);

	pathTrace(devDirectIllum, devIndirectIllum);
	directFilter.temporalAccumulate(devDirectIllum, gBuffer);
	indirectFilter.temporalAccumulate(devIndirectIllum, gBuffer);

	//EAWFilter.filter(devDirectIllum, gBuffer, scene->camera);
	//EAWFilter.filter(devIndirectIllum, gBuffer, scene->camera);

	addImage(devDirectIllum, devIndirectIllum, width, height);
	modulateAlbedo(devDirectIllum, gBuffer);

	if (Settings::ImagePreviewOpt == 2) {
		copyImageToPBO(devPBO, gBuffer.depth(), width, height);
	}
	else if (Settings::ImagePreviewOpt == 3) {
		copyImageToPBO(devPBO, gBuffer.devMotion, width, height);
	}
	else if (Settings::ImagePreviewOpt == 6) {
		copyImageToPBO(devPBO, directFilter.devAccumMoment, width, height);
	}
	else {
		switch (Settings::ImagePreviewOpt) {
		case 0:
			devImage = gBuffer.devAlbedo;
			break;
		case 1:
			devImage = gBuffer.normal();
			break;
		case 4:
			devImage = devDirectIllum;
			break;
		case 5:
			devImage = devIndirectIllum;
			break;
		case 7:
			devImage = directFilter.devAccumColor;
			break;
		}
		copyImageToPBO(devPBO, devImage, width, height, Settings::toneMapping);
	}

	cudaGLUnmapBufferObject(pbo);
	iteration++;
	gBuffer.update(scene->camera);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	Camera& cam = scene->camera;

	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_T:
			Settings::toneMapping = (Settings::toneMapping + 1) % 3;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			cam.position += glm::vec3(0.f, -.1f, 0.f);
			break;
		case GLFW_KEY_SPACE:
			cam.position += glm::vec3(0.f, .1f, 0.f);
			break;
		case GLFW_KEY_R:
			State::camChanged = true;
			break;
		}
	}
}

void mouseScrollCallback(GLFWwindow* window, double offsetX, double offsetY) {
	scene->camera.fov.y -= offsetY;
	scene->camera.fov.y = std::min(scene->camera.fov.y, 45.f);
	State::camChanged = true;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow()) {
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	Camera& cam = scene->camera;

	if (xpos == lastX || ypos == lastY) {
		return; // otherwise, clicking back into window causes re-start
	}

	if (leftMousePressed) {
		// compute new camera parameters
		cam.rotation.x -= (xpos - lastX) / width * 40.f;
		cam.rotation.y += (ypos - lastY) / height * 40.f;
		cam.rotation.y = glm::clamp(cam.rotation.y, -89.9f, 89.9f);
		State::camChanged = true;
	}
	else if (rightMousePressed) {
		float dy = (ypos - lastY) / height;
		cam.position.y += dy;
		State::camChanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.position -= (float)(xpos - lastX) * right * 0.01f;
		cam.position += (float)(ypos - lastY) * forward * 0.01f;
		State::camChanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}