#include "denoiser.h"

__device__ constexpr float Gaussian5x5[] = {
	.0625f, .25f, .375f, .25f, .0625f
};

__global__ void renderGBuffer(DevScene* scene, Camera cam, GBuffer gBuffer) {
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
		bool isLight = scene->materials[intersec.matId].type == Material::Type::Light;
		int matId = intersec.matId;
		if (isLight) {
			matId = NullPrimitive - 1;
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
#endif
		}
		Material material = scene->getTexturedMaterialAndSurface(intersec);

		gBuffer.devAlbedo[idx] = isLight ? glm::vec3(1.f) : material.baseColor;
		gBuffer.normal()[idx] = intersec.norm;
		gBuffer.primId()[idx] = matId;
		gBuffer.depth()[idx] = glm::distance(intersec.pos, ray.origin);

		glm::ivec2 lastPos = gBuffer.lastCamera.getRasterCoord(intersec.pos);
		gBuffer.devMotion[idx] = lastPos.y * cam.resolution.x + lastPos.x;
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
	glm::vec3* devColorOut, glm::vec3* devColorIn, GBuffer gBuffer,
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

	if (primIdP <= NullPrimitive) {
		devColorOut[idxP] = devColorIn[idxP];
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

			if (qx >= cam.resolution.x || qy >= cam.resolution.y) {
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
*/
__global__ void waveletFilter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, float* devVarianceOut, float* devVarainceIn,
	GBuffer gBuffer,
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
		color *= DenoiseCompress;
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


__global__ void temporalAccumulate(
	glm::vec3* devColorAccum, glm::vec2* devMomentAccum, glm::vec3* devColorIn, GBuffer gBuffer, bool first) {
	const float Alpha = .2f;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= gBuffer.width || y >= gBuffer.height) {
		return;
	}
	int idx = y * gBuffer.width + x;

	int primId = gBuffer.primId()[idx];
	int lastIdx = gBuffer.devMotion[idx];

	if (primId <= NullPrimitive || first || gBuffer.primId()[lastIdx] != primId) {
		glm::vec3 c = devColorIn[idx];
		float l = Math::luminance(c);
		devColorAccum[idx] = c;
		devMomentAccum[idx] = { l, l * l };
		return;
	}

	glm::vec3 lastColor = devColorIn[lastIdx];
	float lum = Math::luminance(lastColor);
	devColorAccum[idx] = glm::mix(devColorAccum[idx], lastColor, Alpha);
	devMomentAccum[idx] = glm::mix(devMomentAccum[idx], glm::vec2(lum, lum * lum), Alpha);
}

void GBuffer::create(int width, int height) {
	this->width = width;
	this->height = height;
	int numPixels = width * height;
	devAlbedo = cudaMalloc<glm::vec3>(numPixels);
	devMotion = cudaMalloc<int>(numPixels);
	for (int i = 0; i < 2; i++) {
		devNormal[i] = cudaMalloc<glm::vec3>(numPixels);
		devPrimId[i] = cudaMalloc<int>(numPixels);
		devDepth[i] = cudaMalloc<float>(numPixels);
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
	renderGBuffer<<<blockNum, blockSize>>>(scene, cam, *this);
	checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3* devImage, const GBuffer& gBuffer) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));
	modulate<<<blockNum, blockSize>>>(devImage, gBuffer, gBuffer.width, gBuffer.height);
	checkCUDAError("modulate");
}

void addImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	add<<<blockNum, blockSize>>>(devImage, devIn, width, height);
}

void EAWaveletFilter::filter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		devColorOut, devColorIn, gBuffer, sigDepth, sigNormal, sigLumin, cam, level
	);
	checkCUDAError("EAW Filter");
}

void EAWaveletFilter::filter(
	glm::vec3* devColorOut, glm::vec3* devColorIn,
	float* devVarianceOut, float* devVarianceIn,
	const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		devColorOut, devColorIn, devVarianceOut, devVarianceIn, gBuffer, sigDepth, sigNormal, sigLumin, cam, level
	);
}

void LeveledEAWFilter::create(int width, int height, int level) {
	this->level = level;
	waveletFilter = EAWaveletFilter(width, height);
	devTempImg = cudaMalloc<glm::vec3>(width * height);
}

void LeveledEAWFilter::destroy() {
	cudaSafeFree(devTempImg);
}

void LeveledEAWFilter::filter(glm::vec3*& devColorIn, const GBuffer& gBuffer, const Camera& cam) {
	for (int i = 0; i < level; i++) {
		waveletFilter.filter(devTempImg, devColorIn, gBuffer, cam, i);
		std::swap(devColorIn, devTempImg);
	}
}

void SpatioTemporalFilter::create(int width, int height, int level) {
	this->level = level;
	devAccumColor = cudaMalloc<glm::vec3>(width * height);
	devAccumMoment = cudaMalloc<glm::vec2>(width * height);
	waveletFilter = EAWaveletFilter(width, height);
}

void SpatioTemporalFilter::destroy() {
	cudaSafeFree(devAccumColor);
	cudaSafeFree(devAccumMoment);
}

void SpatioTemporalFilter::temporalAccumulate(glm::vec3* devColorIn, const GBuffer& gBuffer) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));
	::temporalAccumulate<<<blockNum, blockSize>>>(devAccumColor, devAccumMoment, devColorIn, gBuffer, firstTime);
	firstTime = false;
}

void denoiserInit(int width, int height) {
}

void denoiserFree() {
}