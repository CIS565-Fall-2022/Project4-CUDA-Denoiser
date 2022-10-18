#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <device_launch_parameters.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}
	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

///////////////////////////////////////
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void writeToImage(glm::ivec2 resolution,
	int iter, glm::vec3* image, PathSegment* paths
#if _ADAPTIVE_DEBUG_
	, PathSegment* s
#endif
	) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		PathSegment& path = paths[index];

		// contribution based on iter is inherently a bit scuffed
		glm::vec3 pix((image[index] * (float)(iter - 1) + path.color) / (float)iter);
#if _ADAPTIVE_DEBUG_
		// a inverse heatmap to exaggerate greys (not a lot of them though)
		//pix = glm::vec3((_MIN_SPP_ + 1) / (float)paths[index].spp);
		// linear ver. note that likelihood of termination plays a role in heat map.
		pix = glm::vec3((float)paths[index].spp / (float)s->spp);
		image[index] = pix;
#endif
#if _ADAPTIVE_SAMPLING_
		if (!path.skip) {
			image[index] = pix;
			if (path.terminate) {
				path.spp += 1;
				path.colorSum += path.color;
				path.magColorSumSq += glm::length2(path.color);
				if (_MIN_SPP_ < path.spp) {
					// mean and var per color
					float mean2 = glm::length2(path.colorSum / (float)path.spp);
					float variance = path.magColorSumSq / (float)path.spp - mean2;
					if (variance < _PIX_COV_TO_SKIP_) { // pixelIndex covariance or low  enough to assume OK
						path.skip = true; // skip on all future iterations
					}
				}
			}
		}
#else
		image[index] = pix;
#endif
	}
}

// generating kernel to create discrete wavelet transform
__device__ constexpr float generatingKern[5] = { 1.f / 16, 1.f / 4, 3.f / 8, 1.f / 4, 1.f / 16 };
__global__ void aTrousDenoise(glm::ivec2 resolution, glm::vec3* image, GBufferPixel *gBuffer,
	int stride, float colorWeight, float posnWeight, float normWeight) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		// A'Trous Denoising
		glm::vec3 sum(0.f);
		float totalWeight = 0.f;
#pragma unroll
		for (int i = -2; i <= 2; ++i) {
			for (int j = -2; j <= 2; ++j) {
				// Sample surrounding content; stride for empty btwn filter entries
				int sampleIndex = index + (i + resolution.x * j) * stride;
				if (0 <= sampleIndex && sampleIndex < resolution.x * resolution.y) {
					glm::vec3 diff = image[index] - image[sampleIndex];
					float dist = glm::dot(diff, diff);
					float cW = glm::min(glm::exp(-dist / colorWeight), 1.f);

					diff = gBuffer[index].norm - gBuffer[sampleIndex].norm;
					dist = glm::max(glm::dot(diff, diff) / (stride * stride), 0.f);
					float nW = glm::min(glm::exp(-dist / normWeight), 1.f);

					diff = gBuffer[index].posn - gBuffer[sampleIndex].posn;
					dist = glm::dot(diff, diff);
					float pW = glm::min(glm::exp(-dist / posnWeight), 1.f);

					float weight = cW * nW * pW *
						generatingKern[i + 2] * generatingKern[j + 2];
					sum += image[sampleIndex] * weight;
					totalWeight += weight;
				}
			}
		}
		image[index] = sum / totalWeight;
		// End A'Trous Denoising
	}
}


__global__ void bufToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* buffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		// Each thread writes one pixel location in the texture (textel)

		pbo[index].w = 0;
		pbo[index].x = glm::clamp((int)(buffer[index].x * 255.0), 0, 255);
		pbo[index].y = glm::clamp((int)(buffer[index].y * 255.0), 0, 255);
		pbo[index].z = glm::clamp((int)(buffer[index].z * 255.0), 0, 255);
	}
}

__global__ void gBufferFetchAttrib(glm::ivec2 resolution, GBufferPixel* gBuffer,
	int bit, glm::vec3* buffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = bit ?
			gBuffer[index].posn * 0.1f : //scale down posn, o/w wrapping is intense
			gBuffer[index].norm;
		pix = glm::abs(pix);

		buffer[index] = pix;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_temp = NULL; // for misc. things like saving image
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
#if _CACHE_FIRST_BOUNCE_
static ShadeableIntersection* dev_first_isecs = NULL;
#endif

void InitDataContainer(GuiDataContainer* imGuiData) { guiData = imGuiData; }

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_temp, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if _CACHE_FIRST_BOUNCE_
	cudaMalloc(&dev_first_isecs, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_isecs, 0, pixelcount * sizeof(ShadeableIntersection));
#endif
	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_temp);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_gBuffer);
#if _CACHE_FIRST_BOUNCE_
	cudaFree(dev_first_isecs);
#endif
	checkCUDAError("pathtraceFree");
}

// From PBR Book, 13.6.2
// Concentrically maps pts from the [-1, 1) square to the disk centered at 0, 0.
__device__ glm::vec2 sampleDisk(thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(-1, 1);
	glm::vec2 t(u01(rng), u01(rng));
	if (t.x == 0.f && t.y == 0.f) {
		return t; // glm::vec2(0.f);
	}
	float theta, r;
	if (std::abs(t.x) > std::abs(t.y)) {
		r = t.x;
		theta = PI / 4.f * (t.y / t.x);
	}
	else {
		r = t.y;
		theta = PI / 2.f - PI / 4.f * (t.x / t.y);
	}
	return r * glm::vec2(cos(theta), sin(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

#if _ADAPTIVE_SAMPLING_
		if (segment.skip) { // dont do anything for thisone
			segment.remainingBounces = 0;
			return;
		}
#endif
		segment.terminate = false;
		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);

		glm::vec2 jitter(0.f);
		// do not alias or motion blur if caching first bounce for obvious reasons
#if _CACHE_FIRST_BOUNCE_ 
#else
		if (cam.antialias) {
			const float stochasticWeight = 0.5f; // stddev
			thrust::normal_distribution<float> norm(0.f, stochasticWeight);
			jitter = glm::vec2(norm(rng), norm(rng));
		}

		thrust::uniform_real_distribution<float> u01(0, 1);
		segment.ray.time = u01(rng);
#endif
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter.x)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter.y)
		);

		// Depth of field effects, from PBR book, 6.2.3
		if (cam.focalDist > 0 && cam.lensRad > 0) {
			// Set the origin ray to random place on lens (lens as disk).
			glm::vec2 lensRand = cam.lensRad * sampleDisk(rng);
			segment.ray.origin += glm::vec3(lensRand.x, lensRand.y, 0);

			glm::vec3 pFocus = 
				segment.ray.direction * cam.focalDist / std::abs(segment.ray.direction.z);
			segment.ray.direction = glm::normalize(pFocus - glm::vec3(lensRand.x, lensRand.y, 0));
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	Geom* geoms,
	int geoms_size,
	ShadeableIntersection* intersections) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];
#if _ADAPTIVE_SAMPLING_
		if (pathSegment.skip) { return; }
#endif
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool tmp_outside;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++) {
			Geom& geom = geoms[i];
#if _CACHE_FIRST_BOUNCE_
			t = intersectionTest(geom.type, geom.transform, geom.inverseTransform, geom.invTranspose,
				pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
#else
			// motion blur stuff
			if (geom.velocity != glm::vec3(0.f)) {
				glm::mat4 transform = dev_buildTransformationMatrix(
					geom.translation + pathSegment.ray.time * geom.velocity, geom.rotation, geom.scale);
				t = intersectionTest(
					geom.type, transform, glm::inverse(transform), glm::inverseTranspose(transform),
					pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
			else {
				t = intersectionTest(geom.type, geom.transform, geom.inverseTransform, geom.invTranspose,
					pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
#endif		
			// Compute minimum t from intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				outside = tmp_outside;
			}
		}

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		}
		else {
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].outside = outside;
		}
	}
}

// Shade path segments based on intersections, generate new rays w/ BSDF.
__global__ void shadeMaterial (
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int depth) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
#if _STREAM_COMPACTION_
#else // below skips if .skip = true (see generateRay)
	if (pathSegments[idx].remainingBounces == 0) { return; }
#endif
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			Material material = materials[intersection.materialId];
			if (material.emittance > 0.0f) { // material is bright; "light" the ray
				pathSegments[idx].terminate = true;
				pathSegments[idx].color *= (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0; // terminate if hit light
			} else {
				// BSDF evaluation: in-place assigns new direction + color for ray
				scatterRay(pathSegments[idx],
					getPointOnRay(pathSegments[idx].ray, intersection.t),
					intersection.surfaceNormal,
					intersection.outside,
					material,
					makeSeededRandomEngine(iter, idx, depth));
				if (--pathSegments[idx].remainingBounces == 0) {
					pathSegments[idx].color = BACKGROUND_COLOR;
				}
			}
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegments[idx].color = BACKGROUND_COLOR;
			pathSegments[idx].remainingBounces = 0; // terminate if hit nothing
		}
	}
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		gBuffer[idx].norm = shadeableIntersections[idx].surfaceNormal;
		gBuffer[idx].posn = pathSegments[idx].ray.origin + 
			shadeableIntersections[idx].t * pathSegments[idx].ray.direction;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////
	// --- Start generate array of path rays (that come out of the camera)
	//   * Each path ray color starts as white = (1, 1, 1).
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	// --- End generate rays
	
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	bool iterationComplete = false;
	while (!iterationComplete) {
		// --- start compute intersection ---
	//   * Compute an intersection in the scene for each path ray.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if _CACHE_FIRST_BOUNCE_
		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections,
				dev_first_isecs,
				pixelcount * sizeof(ShadeableIntersection),
				cudaMemcpyDeviceToDevice);
		} else {
#endif
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections);
		checkCUDAError("trace one bounce");
#if _CACHE_FIRST_BOUNCE_
		}
		if (iter == 1 && depth == 0) {
			cudaMemcpy(dev_first_isecs,
				dev_intersections,
				pixelcount * sizeof(ShadeableIntersection),
				cudaMemcpyDeviceToDevice);
		}
#endif
		cudaDeviceSynchronize();
		if (depth == 0) {
			generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		depth++;
		// --- end compute intersection --- 

#if _GROUP_RAYS_BY_MATERIAL_
		thrust::sort_by_key(thrust::device,
			dev_intersections, dev_intersections + num_paths,
			dev_paths,
			compare_intersection_mat());
#endif
		// can't use a 2D kernel launch any more - switch to 1D.
		// --- begin Shading Stage ---
		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth);
		// --- end shading stage
		
		// --- begin stream compaction for all of the terminated paths
		//	   determine if iterations should end (ie all paths done)
#if _STREAM_COMPACTION_
		dev_path_end = thrust::partition(thrust::device,
			dev_paths, dev_path_end,
			partition_terminated_paths()); // overloaded struct; lambdas need special compilation flag
		num_paths = dev_path_end - dev_paths;
		iterationComplete = (num_paths == 0);
#else
		iterationComplete = (depth == traceDepth);
#endif
		// --- end stream compaction

		if (guiData != NULL) {
			guiData->TracedDepth = depth;
		}
	}

#if _ADAPTIVE_DEBUG_
	PathSegment* s = thrust::max_element(thrust::device, dev_paths, dev_paths + pixelcount, compare_path_spp());
	writeToImage<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, iter, dev_image, dev_paths, s);
#else 
	writeToImage<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, iter, dev_image, dev_paths);
#endif
}

// bit = 0 for norm, 1 for posn. better as enum
void showGBuffer(uchar4* pbo, int bit, bool save) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	gBufferFetchAttrib<<<blocksPerGrid2d, blockSize2d>>>
		(cam.resolution, dev_gBuffer, bit, dev_temp);

	if (!save) {
		bufToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_temp);
	} else {
		bufToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_temp);
		const int pixelcount = cam.resolution.x * cam.resolution.y;
		cudaMemcpy(hst_scene->state.image.data(), dev_temp,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}
}

// timing for analysis stuff
#if _TIME_ATROUS_DENOISER_
#include <chrono>
#endif

void showImage(uchar4* pbo, bool denoise, int filterSize, float colorWeight, float posWeight, float normWeight, bool save) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// if denoising, use dev_temp for stuff
	if (denoise) {
		cudaMemcpy(dev_temp, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
		
#if _TIME_ATROUS_DENOISER_
		auto start = chrono::high_resolution_clock::now();
#endif
		for (int f = 0; f < filterSize; ++f) {
			aTrousDenoise<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, dev_temp, dev_gBuffer,
				1 << f, colorWeight, posWeight, normWeight);
		}
#if _TIME_ATROUS_DENOISER_
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		cout << duration.count() << endl;
#endif
		bufToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_temp);
		if (save) {
			cudaMemcpy(hst_scene->state.image.data(), dev_temp, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		}
	} else {
		bufToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_image);
		if (save) {
			cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		}
	}
}