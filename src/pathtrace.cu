#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <chrono>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
using namespace std::chrono;

//option
#define CACHE_FIRST_INTERSECTION 1
#define MATERIAL_CONTIGUOUS 0
#define ANTIALIASING 1
#define DEPTH_OF_FIELD 0

#define ENABLE_CACHE_FIRST_INTERSECTION (CACHE_FIRST_INTERSECTION && !ANTIALIASING && !DEPTH_OF_FIELD)

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//for stream compaction
struct shouldContinue
{
	__host__ __device__
		bool operator()(const PathSegment x)
	{
		bool stop = x.remainingBounces <= 0 || (x.color.r < EPSILON&& x.color.b < EPSILON&& x.color.g < EPSILON);
		return !stop;
	}
};

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, int type) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		if (type == 0) {
			pbo[index].w = 0;
			pbo[index].x = gBuffer[index].normal.x * 128 + 128;
			pbo[index].y = gBuffer[index].normal.y * 128 + 128;
			pbo[index].z = gBuffer[index].normal.z * 128 + 128;
		}
		else if (type == 1) {
			float timeToIntersect = gBuffer[index].t * 256.0;

			pbo[index].w = 0;
			pbo[index].x = timeToIntersect;
			pbo[index].y = timeToIntersect;
			pbo[index].z = timeToIntersect;
		}
		else if (type == 2) {
			glm::vec3 scalePos = gBuffer[index].position / 15.f * 255.f;
			pbo[index].w = 0;
			pbo[index].x = scalePos.x;
			pbo[index].y = scalePos.y;
			pbo[index].z = scalePos.z;
		}
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
#if ENABLE_CACHE_FIRST_INTERSECTION
static ShadeableIntersection* dev_cacheIntersections = NULL;
#endif // ENABLE_CACHE_FIRST_INTERSECTION
static glm::vec3* dev_denoiseBuffer = NULL;
static float* dev_denoiseKernel = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need
#if ENABLE_CACHE_FIRST_INTERSECTION
	cudaMalloc(&dev_cacheIntersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cacheIntersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif // ENABLE_CACHE_FIRST_INTERSECTION

	cudaMalloc(&dev_denoiseBuffer, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoiseBuffer, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_denoiseKernel, 25 * sizeof(float));
	float tmpDenoiseKernel1D[] = { 1.f / 16.f, 1.f / 4.f, 3.f / 8.f, 1.f / 4.f, 1.f / 16.f };
	float tmpDenoiseKernel2D[5][5];
	for (size_t i = 0; i < 5; ++i) {
		for (size_t j = 0; j < 5; ++j) {
			tmpDenoiseKernel2D[i][j] = tmpDenoiseKernel1D[i] * tmpDenoiseKernel1D[j];
		}
	}
	cudaMemcpy(dev_denoiseKernel, tmpDenoiseKernel2D, sizeof(float) * 5 * 5, cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
#if ENABLE_CACHE_FIRST_INTERSECTION
	cudaFree(dev_cacheIntersections);
#endif // ENABLE_CACHE_FIRST_INTERSECTION
	cudaFree(dev_denoiseBuffer);
	cudaFree(dev_denoiseKernel);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;


    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, cam.resolution.y * cam.resolution.x - index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        float pixelx = x, pixely = y;
        glm::mat4 cameraToWorld(glm::vec4(cam.right, 0), glm::vec4(cam.up, 0), glm::vec4(cam.view, 0), glm::vec4(cam.position, 1));

#if ANTIALIASING
        pixelx = x + u01(rng) - 0.5;
        pixely = y + u01(rng) - 0.5;
#endif

#if DEPTH_OF_FIELD
        float phi, r, u, v;
        r = sqrt(u01(rng));
        phi = TWO_PI * u01(rng);
        u = r * cos(phi);
        v = r * sin(phi);
        glm::vec3 pLens = cam.lensRadius * glm::vec3(u, v, 0);
        glm::vec3 pPixel = glm::vec3(-cam.pixelLength.x * (pixelx - (float)cam.resolution.x * 0.5f), -cam.pixelLength.y * (pixely - (float)cam.resolution.y * 0.5f), 1);
        glm::vec3 pFocus = cam.focalDistance * pPixel;
        segment.ray.origin = glm::vec3(cameraToWorld * glm::vec4(pLens, 1));
        segment.ray.direction = glm::normalize(glm::mat3(cameraToWorld) * (pFocus - pLens));
#else
        glm::vec3 pPixel = glm::vec3(-cam.pixelLength.x * (pixelx - (float)cam.resolution.x * 0.5f), -cam.pixelLength.y * (pixely - (float)cam.resolution.y * 0.5f), 1);
        segment.ray.direction = glm::mat3(cameraToWorld) * pPixel;
        /*segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (pixelx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (pixely - (float)cam.resolution.y * 0.5f)
        );*/
#endif
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeSimpleMaterials (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, int depth
	, GBufferPixel* gbuffer
	)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
			if (depth == 1) {
				gbuffer[idx].color = materialColor;
			}

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;		//stop when hit light source
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 isectPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], isectPoint, intersection.surfaceNormal, material, rng);
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
	gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
	gBuffer[idx].position = shadeableIntersections[idx].t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
  }
}

__global__ void setGbufferColor(int pixelCount, glm::vec3* image, GBufferPixel* gbuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pixelCount) {
		gbuffer[idx].color = image[idx];
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void denoiseNaive(glm::vec3* image, glm::vec3* buffer, float* kernel, int pixelCount, glm::ivec2 resolution, int distance) {
	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = blockDim.y * blockIdx.y + threadIdx.y;
	int index = idxY * resolution.y + idxX;
	if (idxX < resolution.x && idxY < resolution.y) {
		float totalWeight = 0;
		glm::vec3 color(0.f);

		for (int dy = -2; dy <= 2; ++dy) {
			int y = idxY + dy * distance;
			if (y < 0 || y >= resolution.y) {
				continue;
			}

			for (int dx = -2; dx <= 2; ++dx) {
				int x = idxX + dx * distance;
				if (x < 0 || x >= resolution.x) {
					continue;
				}

				float weight = kernel[(dy + 2) * 5 + dx + 2];
				color += image[y * resolution.y + x] * weight;
				totalWeight += weight;
			}
		}
		if (totalWeight == 0) {
			color = glm::vec3(0.f);
		}
		else {
			color /= totalWeight;
		}
		buffer[index] = color;
	}
}

__device__ float computeLogWeight(glm::vec3& v0, glm::vec3& v1, float sigma2) {
	return sigma2 == 0 ? 0 : -glm::length(v1 - v0) / sigma2;
}

__global__ void denoiseEdgeAvoiding(GBufferPixel* gBuffer, DenoiseParm dParm, glm::vec3* image, glm::vec3* buffer, float* kernel, int pixelCount, glm::ivec2 resolution, int distance) {
	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = blockDim.y * blockIdx.y + threadIdx.y;
	int index = idxY * resolution.y + idxX;
	if (idxX < resolution.x && idxY < resolution.y) {
		float totalWeight = 0;
		glm::vec3 color(0.f);
		GBufferPixel p0 = gBuffer[index];

		for (int dy = -2; dy <= 2; ++dy) {
			int y = idxY + dy * distance;
			if (y < 0 || y >= resolution.y) {
				continue;
			}

			for (int dx = -2; dx <= 2; ++dx) {
				int x = idxX + dx * distance;
				if (x < 0 || x >= resolution.x) {
					continue;
				}
				GBufferPixel p1 = gBuffer[y * resolution.y + x];

				float weight = kernel[(dy + 2) * 5 + dx + 2];
				float logColorWeight = computeLogWeight(p0.color, p1.color, dParm.colorWeight * dParm.colorWeight);
				float logNormalWeight = computeLogWeight(p0.normal, p1.normal, dParm.normalWeight * dParm.normalWeight);
				float logPosWeight = computeLogWeight(p0.position, p1.position, dParm.positionWeight * dParm.positionWeight);
				weight *= __expf(logColorWeight + logNormalWeight + logPosWeight);
				color += image[y * resolution.y + x] * weight;
				totalWeight += weight;
			}
		}
		if (totalWeight == 0) {
			color = glm::vec3(0.f);
		}
		else {
			color /= totalWeight;
		}
		buffer[index] = color;
	}
}

void denoise(int pixelCount, glm::ivec2 resolution, DenoiseParm dParm, dim3 blockSize2d, dim3 blocksPerGrid2d) {
	for (int distance = 1; distance <= dParm.filterSize; distance *= 2) {
		if (dParm.denoise == 1) {
			denoiseNaive << <blocksPerGrid2d, blockSize2d >> > (dev_image, dev_denoiseBuffer, dev_denoiseKernel, pixelCount, resolution, distance);
		}
		else {
			denoiseEdgeAvoiding << <blocksPerGrid2d, blockSize2d >> > (dev_gBuffer, dParm, dev_image, dev_denoiseBuffer, dev_denoiseKernel, pixelCount, resolution, distance);
		}
		glm::vec3* tmp = dev_image;
		dev_image = dev_denoiseBuffer;
		dev_denoiseBuffer = tmp;
	}
	
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, DenoiseParm dParm) {
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

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  	// Empty gbuffer
  	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	auto start = system_clock::now();
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if ENABLE_CACHE_FIRST_INTERSECTION
		if (depth == 0 && iter != 1) {
			cudaMemcpy(dev_intersections, dev_cacheIntersections, sizeof(ShadeableIntersection) * num_paths, cudaMemcpyDeviceToDevice);
			checkCUDAError("loadIntersections");
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");

			if (depth == 0 && iter == 1) {
				cudaMemcpy(dev_cacheIntersections, dev_intersections, sizeof(ShadeableIntersection) * num_paths, cudaMemcpyDeviceToDevice);
				checkCUDAError("cacheIntersections");
			}
		}

#else
		// tracing
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
#endif // CACHE_FIRST_INTERSECTION

		cudaDeviceSynchronize();
		if (depth == 0) {
    		generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
  		}
		depth++;

		//thrust ptr
		thrust::device_ptr<PathSegment> thrust_dev_paths(dev_paths);
		thrust::device_ptr<ShadeableIntersection> thrust_dev_intersection(dev_intersections);

		auto start = system_clock::now();
#if MATERIAL_CONTIGUOUS
		thrust::sort_by_key(thrust_dev_intersection, thrust_dev_intersection + num_paths, thrust_dev_paths);
#endif // MATERIAL_CONTIGUOUS
		auto end = system_clock::now();
		if (iter == 10) {
			//cout << "sort: " << duration_cast<microseconds>(end - start).count() << endl;
		}

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		start = system_clock::now();
		shadeSimpleMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth,
			dev_gBuffer
			);
		end = system_clock::now();
		if (iter == 10) {
			//cout << "shade: " << duration_cast<microseconds>(end - start).count() << endl;
		}

		start = system_clock::now();
		thrust::device_ptr<PathSegment> thrust_dev_paths_end = thrust::partition(thrust_dev_paths, thrust_dev_paths + num_paths, shouldContinue());
		end = system_clock::now();
		if (iter == 10) {
			//cout << "partition: " << duration_cast<microseconds>(end - start).count() << endl;
		}
		dev_path_end = thrust_dev_paths_end.get();
		num_paths = dev_path_end - dev_paths;
		iterationComplete = depth >= traceDepth || num_paths == 0; // TODO: should be based off stream compaction results.

	}
	auto end = system_clock::now();
	cout << "shading: " << duration_cast<microseconds>(end - start).count() << endl;


	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	//setGbufferColor << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_gBuffer);
	start = system_clock::now();
	if (dParm.denoise) {
		denoise(pixelcount, cam.resolution, dParm, blockSize2d, blocksPerGrid2d);
	}
	end = system_clock::now();
	cout << "denoise: " << duration_cast<microseconds>(end - start).count() << endl;



    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, 2);
}

void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}
