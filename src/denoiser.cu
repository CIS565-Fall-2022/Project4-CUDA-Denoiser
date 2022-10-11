#include "denoiser.h"

constexpr float Gaussian5x5Weight[] = {
	.0625f, .25f, .375f, .25f, .0625f
};

__global__ void renderGBufferKern(DevScene* scene, Camera cam, GBuffer gBuffer) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idx = y * cam.resolution.x + x;

	float aspect = float(cam.resolution.x) / cam.resolution.y;
	float tanFovY = glm::tan(glm::radians(cam.fov.y));
	glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
	glm::vec2 scr = glm::vec2(x, y) * pixelSize;
	glm::vec2 ruv = scr + pixelSize * glm::vec2(.5f);
	ruv = 1.f - ruv * 2.f;

	glm::vec3 pLens(0.f);
	glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
	glm::vec3 dir = pFocusPlane - pLens;

	Ray ray;
	ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
	ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;

	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId != NullPrimitive) {
		if (scene->materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
#endif
		}
		Material material = scene->getTexturedMaterialAndSurface(intersec);

		gBuffer.devAlbedo[idx] = material.baseColor;
		gBuffer.devNormal[idx] = intersec.norm;
		gBuffer.devPrimId[idx] = intersec.primId;
	}
	else {
		gBuffer.devAlbedo[idx] = glm::vec3(0.f);
		gBuffer.devNormal[idx] = glm::vec3(0.f);
		gBuffer.devPrimId[idx] = NullPrimitive;
	}
}

__device__ float weightLuminance(glm::vec3* color, glm::ivec2 p, glm::ivec2 q) {
	return 0.f;
}

__device__ float weightLuminance(glm::vec4* colorVar, glm::ivec2 p, glm::ivec2 q) {
	return 0.f;
}

template<int Level>
__global__ void EAWaveletFilter(
	GBuffer* devGBuffer, glm::vec3* devColorOut, glm::vec3* devColorIn,
	float sigDepth, float sigNormal, float sigLuminance,
	int width, int height
) {
	constexpr int Step = 1 << Level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= -2; j++) {
		}
	}
}


/*
* SVGF version, filtering variance at the same time
* Variance is stored as the last component of vec4
*/
template<int Level>
__global__ void EAWaveletFilter(
	GBuffer* devGBuffer, glm::vec4* devColorVarOut, glm::vec4* devColorVarIn,
	float sigDepth, float sigNormal, float sigLuminance,
	int width, int height
) {
	constexpr int Step = 1 << Level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= -2; j++) {
		}
	}
}

__global__ void modulate(glm::vec3* devImage, GBuffer gBuffer, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		devImage[idx] = devImage[idx] * glm::max(gBuffer.devAlbedo[idx] - DEMODULATE_EPS, glm::vec3(0.f));
	}
}

void GBuffer::create(int width, int height) {
	int numPixels = width * height;
	cudaMalloc(&devAlbedo, numPixels * sizeof(glm::vec3));
	cudaMalloc(&devNormal, numPixels * sizeof(glm::vec3));
	cudaMalloc(&devPrimId, numPixels * sizeof(int));
}

void GBuffer::destroy() {
	cudaSafeFree(devAlbedo);
	cudaSafeFree(devNormal);
	cudaSafeFree(devPrimId);
}

void denoiserInit(int width, int height) {
}

void denoiserFree() {
}

void renderGBuffer(DevScene* scene, Camera cam, GBuffer gBuffer) {
	constexpr int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(cam.resolution.x, BlockSize), ceilDiv(cam.resolution.y, BlockSize));
	renderGBufferKern<<<blockNum, blockSize>>>(scene, cam, gBuffer);
	checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3* devImage, GBuffer gBuffer, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	modulate<<<blockNum, blockSize>>>(devImage, gBuffer, width, height);
	checkCUDAError("modulate");
}