#include "denoiser.h"

__device__ constexpr float Gaussian5x5[] = {
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
		int matId = intersec.matId;
		if (scene->materials[intersec.matId].type == Material::Type::Light) {
			matId = NullPrimitive;
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
#endif
		}
		Material material = scene->getTexturedMaterialAndSurface(intersec);

		gBuffer.devAlbedo[idx] = material.baseColor;
		gBuffer.normal()[idx] = intersec.norm;
		gBuffer.primId()[idx] = matId;
		gBuffer.depth()[idx] = glm::distance(intersec.pos, ray.origin);
	}
	else {
		gBuffer.devAlbedo[idx] = glm::vec3(0.f);
		gBuffer.normal()[idx] = glm::vec3(0.f);
		gBuffer.primId()[idx] = NullPrimitive;
		gBuffer.depth()[idx] = 0.f;
	}
}

__device__ float weightLuminance(glm::vec3* color, int p, int q) {
	return 0.f;
}

__device__ float weightLuminance(glm::vec4* colorVar, int p, int q) {
	return 0.f;
}

__device__ float weightNormal(const GBuffer& gBuffer, int p, int q) {

}

__global__ void waveletFilter(
	GBuffer gBuffer, glm::vec3* devColorOut, glm::vec3* devColorIn,
	float sigDepth, float sigNormal, float sigLuminance, Camera cam, int level
) {
	int step = 1 << level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idxP = y * cam.resolution.x + x;
	int primIdP = gBuffer.primId()[idxP];

	if (primIdP == NullPrimitive) {
		return;
	}

	glm::vec3 normP = gBuffer.normal()[idxP];
	glm::vec3 colorP = devColorIn[idxP];
	glm::vec3 posP = cam.getPosition(x, y, gBuffer.depth()[idxP]);

	glm::vec3 sum(0.f);
	float sumWeight = 0.f;
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			int qx = x + i * step;
			int qy = y + j * step;
			int idxQ = qy * cam.resolution.x + qx;

			if (idxQ < 0 || idxQ >= cam.resolution.x * cam.resolution.y) {
				continue;
			}
			if (gBuffer.primId()[idxQ] != primIdP) {
				continue;
			}
			glm::vec3 normQ = gBuffer.normal()[idxQ];
			glm::vec3 colorQ = devColorIn[idxQ];
			glm::vec3 posQ = cam.getPosition(qx, qy, gBuffer.depth()[idxQ]);

			float distColor2 = glm::dot(colorP - colorQ, colorP - colorQ);
			float wColor = glm::min(1.f, glm::exp(-distColor2 / sigLuminance));

			float distNorm2 = glm::dot(normP - normQ, normP - normQ);
			float wNorm = glm::min(1.f, glm::exp(-distNorm2 / sigNormal));

			float distPos2 = glm::dot(posP - posQ, posP - posQ);
			float wPos = glm::min(1.f, glm::exp(-distPos2 / sigDepth));

			float weight = wColor * wNorm * wPos * Gaussian5x5[i + 2] * Gaussian5x5[j + 2];
			sum += colorQ * weight;
			sumWeight += weight;
		}
	}
	devColorOut[idxP] = (sumWeight == 0.f) ? devColorIn[idxP] : sum / sumWeight;
}


/*
* SVGF version, filtering variance at the same time
* Variance is stored as the last component of vec4
*/
__global__ void waveletFilter(
	GBuffer gBuffer, glm::vec4* devColorVarOut, glm::vec4* devColorVarIn,
	float sigDepth, float sigNormal, float sigLuminance, Camera cam, int level
) {
	int step = 1 << level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
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
		glm::vec3 color = devImage[idx];
		color = color / (1.f - color);
		//color *= DenoiseCompress;
		devImage[idx] = color * glm::max(gBuffer.devAlbedo[idx]/* - DEMODULATE_EPS*/, glm::vec3(0.f));
	}
}

__global__ void add(glm::vec3* devImage, glm::vec3* devIn, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		devImage[idx] += devIn[idx];
	}
}

void GBuffer::create(int width, int height) {
	int numPixels = width * height;
	cudaMalloc(&devAlbedo, numPixels * sizeof(glm::vec3));
	cudaMalloc(&devMotion, numPixels * sizeof(glm::vec2));
	for (int i = 0; i < 2; i++) {
		cudaMalloc(&devNormal[i], numPixels * sizeof(glm::vec3));
		cudaMalloc(&devPrimId[i], numPixels * sizeof(int));
		cudaMalloc(&devDepth[i], numPixels * sizeof(float));
	}
}

void GBuffer::destroy() {
	cudaSafeFree(devAlbedo);
	cudaSafeFree(devMotion);
	for (int i = 0; i < 2; i++) {
		cudaSafeFree(devNormal[i]);
		cudaSafeFree(devPrimId[i]);
		cudaSafeFree(devDepth[i]);
	}
}

void GBuffer::update(const Camera& cam) {
	lastCamera = cam;
	frame ^= 1;
}

void GBuffer::render(DevScene* scene, const Camera& cam) {
	constexpr int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(cam.resolution.x, BlockSize), ceilDiv(cam.resolution.y, BlockSize));
	renderGBufferKern<<<blockNum, blockSize>>>(scene, cam, *this);
	checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3* devImage, GBuffer gBuffer, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	modulate<<<blockNum, blockSize>>>(devImage, gBuffer, width, height);
	checkCUDAError("modulate");
}

void composeImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	add<<<blockNum, blockSize>>>(devImage, devIn, width, height);
}

void EAWaveletFilter::filter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		gBuffer, devColorOut, devColorIn, sigDepth, sigNormal, sigLumin, cam, level
	);
	checkCUDAError("EAW Filter");
}

void EAWaveletFilter::filter(
	glm::vec4* devColorVarOut, glm::vec4* devColorVarIn, const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		gBuffer, devColorVarOut, devColorVarIn, sigDepth, sigNormal, sigLumin, cam, level
	);
}


void denoiserInit(int width, int height) {
}

void denoiserFree() {
}