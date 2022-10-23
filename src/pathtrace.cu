#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bvhTree.h"

#define ERRORCHECK 1

#define SORT_BY_MATERIAL 0
#define ANTIALIASING 1
#define CACHE_FIRST_INTERSECTION (1 && !ANTIALIASING)
#define DENOISE_TIME 1
#define EDGE_AVOIDING 1


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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void getImageColor(glm::ivec2 resolution, int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
		color.x = glm::clamp((pix.x / iter), 0.f, 1.f);
		color.y = glm::clamp((pix.y / iter), 0.f, 1.f);
		color.z = glm::clamp((pix.z / iter), 0.f, 1.f);

		image[index] = color;
	}
}

__global__ void getImageColorSum(glm::ivec2 resolution, int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
		color.x = pix.x * iter;
		color.y = pix.y * iter;
		color.z = pix.z * iter;

		image[index] = color;
	}
}


__global__ void sendDenoisedImageToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = pix.x * 255.0;
		pbo[index].y = pix.y * 255.0;
		pbo[index].z = pix.z * 255.0;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_outDenoisedImage = NULL;
static glm::vec3* dev_inDenoisedImage = NULL;
static Geom* dev_geoms = NULL;

#if USE_BVH_FOR_INTERSECTION
static BVHNode* dev_bvhNodes = NULL;
static int bvhNodes_size = 0;
#else
static Triangle* dev_faces = NULL;
#endif

static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

#if DENOISE_TIME
static cudaEvent_t denoiseStart = NULL;
static cudaEvent_t denoiseEnd = NULL;
static float denoiseTime = 0.f;
#endif

#if CACHE_FIRST_INTERSECTION
static ShadeableIntersection* dev_intersectionsCache = NULL;
#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

#if USE_BVH_FOR_INTERSECTION
	BVHTree bvhTree;
	bvhTree.build(hst_scene->faces);
	bvhNodes_size = bvhTree.bvhNodes.size();
#endif

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_outDenoisedImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_outDenoisedImage, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_inDenoisedImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_inDenoisedImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if USE_BVH_FOR_INTERSECTION
	cudaMalloc(&dev_bvhNodes, bvhTree.bvhNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhNodes, bvhTree.bvhNodes.data(), bvhTree.bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
#else
	cudaMalloc(&dev_faces, scene->faces.size() * sizeof(Triangle));
	cudaMemcpy(dev_faces, scene->faces.data(), scene->faces.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
#if CACHE_FIRST_INTERSECTION
	cudaMalloc(&dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersectionsCache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if DENOISE_TIME
	cudaEventCreate(&denoiseStart);
	cudaEventCreate(&denoiseEnd);
#endif

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_outDenoisedImage);
	cudaFree(dev_inDenoisedImage);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if CACHE_FIRST_INTERSECTION
	cudaFree(dev_intersectionsCache);
#endif

#if USE_BVH_FOR_INTERSECTION
	cudaFree(dev_bvhNodes);
#else
	cudaFree(dev_faces);
#endif

	cudaFree(dev_gBuffer);

#if DENOISE_TIME
	if (denoiseStart) cudaEventDestroy(denoiseStart);
	if (denoiseEnd) cudaEventDestroy(denoiseEnd);
#endif

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

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#if ANTIALIASING
		thrust::uniform_real_distribution<float> shiftDistGenerator(-0.5f, 0.5f);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index << 1, 0);
		float dx = shiftDistGenerator(rng);
		rng = makeSeededRandomEngine(iter, index << 1 + 1, 0);
		float dy = shiftDistGenerator(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + dx - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + dy - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
#if USE_BVH_FOR_INTERSECTION
	, BVHNode* bvhNodes
	, int bvhNodes_size
#else
	, Triangle* faces
	, int faces_size
#endif
	, ShadeableIntersection* intersections
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

#if USE_BVH_FOR_INTERSECTION
		bool didBVHIntersection = false;
#endif

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
#if USE_BVH_FOR_INTERSECTION
			else if (!didBVHIntersection && geom.type == MESH)
			{
				didBVHIntersection = true;
				t = bvhIntersectionTest(geoms, bvhNodes, bvhNodes_size, pathSegment.ray, tmp_intersect, tmp_normal, outside, &hit_geom_index);
			}
			else if (didBVHIntersection && geom.type == MESH)
			{
				continue;
			}
#else
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, faces, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
#endif

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
#if USE_BVH_FOR_INTERSECTION
				if (geom.type != MESH) hit_geom_index = i;
#else
				hit_geom_index = i;
#endif
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

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
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

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
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

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths || pathSegments[idx].remainingBounces <= 0) return;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	// If there was no intersection, color the ray black.
	if (intersection.t <= 0.0f)
	{
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
		return;
	}

	// if the intersection exists...
	Material material = materials[intersection.materialId];

	// If the material indicates that the object was a light, "light" the ray
	if (material.emittance > 0.0f)
	{
		pathSegments[idx].color *= (material.color * material.emittance);
		pathSegments[idx].remainingBounces = 0;
	}
	// Otherwise, compute the next bounce ray
	else
	{
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
		scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
		pathSegments[idx].remainingBounces--;
	}
}


__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].t = shadeableIntersections[idx].t;
		gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
		gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct isRayBouncing
{
	__host__ __device__
		bool operator()(const PathSegment& pathSegment)
	{
		return pathSegment.remainingBounces > 0;
	}
};

struct isSmallerMaterialId
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& intersection1, const ShadeableIntersection& intersection2)
	{
		return intersection1.materialId < intersection2.materialId;
	}
};

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

	PathSegment* dev_pathsStart = dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_INTERSECTION
		if (depth == 0 && iter != 1)
		{
			cudaMemcpy(dev_intersections, dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
#if USE_BVH_FOR_INTERSECTION
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_bvhNodes, bvhNodes_size, dev_intersections);
#else
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_faces, hst_scene->faces.size(), dev_intersections);
#endif
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}

		if (depth == 0 && iter == 1)
		{
			cudaMemcpy(dev_intersectionsCache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#else
#if USE_BVH_FOR_INTERSECTION
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_bvhNodes, bvhNodes_size, dev_intersections);
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_faces, hst_scene->faces.size(), dev_intersections);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif


		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
#if SORT_BY_MATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, isSmallerMaterialId());
#endif

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, dev_intersections, dev_paths, dev_materials);

		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, isRayBouncing());

		num_paths = dev_path_end - dev_paths;

		depth++;
		iterationComplete = num_paths == 0;
	}

	// Assemble this iteration and apply it to the image
	//dev_paths = thrust::raw_pointer_cast(thrust_paths);
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	//sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);


	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}


__global__ void kernDenoise(glm::vec3* outImage, glm::vec3* inImage, glm::ivec2 resolution, GBufferPixel* gBuffer,
	float stdColorWeight, float stdNormalWeight, float stdPosWeight, float stepWidth) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= resolution.x || y >= resolution.y) return;
	int idx = x + y * resolution.x;

	const float kernel[25] =
	{
		0.0030,    0.0133,    0.0219,    0.0133,    0.0030,
		0.0133,    0.0596,    0.0983,    0.0596,    0.0133,
		0.0219,    0.0983,    0.1621,    0.0983,    0.0219,
		0.0133,    0.0596,    0.0983,    0.0596,    0.0133,
		0.0030,    0.0133,    0.0219,    0.0133,    0.0030
	};

	glm::vec3 currColor = inImage[idx];
	glm::vec3 currNor = gBuffer[idx].nor;
	glm::vec3 currPos = gBuffer[idx].pos;

	glm::vec3 sum;
	float cumulativeWeight = 0.f;

	for (int i = -2; i <= 2; i++)
	{
		for (int j = -2; j <= 2; j++)
		{
			glm::ivec2 uv = glm::ivec2(x + i * stepWidth, y + j * stepWidth);
			uv = glm::clamp(uv, glm::ivec2(0, 0), resolution - 1);
			int sampleIdx = uv.x + uv.y * resolution.x;
			
			glm::vec3 sampleColor = inImage[sampleIdx];
#if EDGE_AVOIDING
			glm::vec3 t = currColor - sampleColor;

			float dist2 = glm::dot(t, t);
			float colorWeight = min(exp(-(dist2) / stdColorWeight), 1.f);

			glm::vec3 sampleNor = gBuffer[sampleIdx].nor;
			t = currNor - sampleNor;
			dist2 = max(glm::dot(t, t) / (stepWidth * stepWidth), 0.f);
			float norWeight = min(exp(-(dist2) / stdNormalWeight), 1.f);

			glm::vec3 samplePos = gBuffer[sampleIdx].pos;
			t = currPos - samplePos;
			dist2 = glm::dot(t, t);
			float posWeight = min(exp(-(dist2) / stdPosWeight), 1.f);

			float weight = colorWeight * norWeight * posWeight;
			int kernelIdx = (j + 2) * 5 + i + 2;
			sum += sampleColor * weight * kernel[kernelIdx];
			cumulativeWeight += weight * kernel[kernelIdx];
#else
			int kernelIdx = (j + 2) * 5 + i + 2;
			sum += sampleColor * kernel[kernelIdx];
			cumulativeWeight += kernel[kernelIdx];
#endif
		}
	}
	outImage[idx] = sum / cumulativeWeight;
}


void denoise(int iter, int filterSize, float colorWeight, float norWeight, float posWeight)
{
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	cudaMemcpy(dev_inDenoisedImage, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	getImageColor << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_inDenoisedImage);


#if DENOISE_TIME
	denoiseTime = 0.f;
	cudaEventRecord(denoiseStart);
#endif   

	int stepWidth = 1;
	int blurIter = glm::ceil(glm::log2(filterSize / 2));

	for (int i = 0; i < blurIter; i++)
	{
		kernDenoise << <blocksPerGrid2d, blockSize2d >> > (dev_outDenoisedImage, dev_inDenoisedImage, cam.resolution, dev_gBuffer,
			colorWeight, norWeight, posWeight, stepWidth);
		std::swap(dev_outDenoisedImage, dev_inDenoisedImage);
		stepWidth <<= 1;
	}
	std::swap(dev_outDenoisedImage, dev_inDenoisedImage);

#if DENOISE_TIME
	cudaEventRecord(denoiseEnd);
	cudaEventSynchronize(denoiseEnd);
	float elapsedTime = 0.f;
	cudaEventElapsedTime(&elapsedTime, denoiseStart, denoiseEnd);
	denoiseTime += elapsedTime;
#endif

	cudaMemcpy(dev_inDenoisedImage, dev_outDenoisedImage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	getImageColorSum << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_inDenoisedImage);
	cudaMemcpy(hst_scene->state.image.data(), dev_inDenoisedImage,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}


__global__ void gbufferTToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer[index].t * 255.f * 0.05f;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;
	}
}

__global__ void gbufferNorToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 color = (gBuffer[index].nor + 1.f) * 0.5f * 255.f;

		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void gbufferPosToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 color = (glm::normalize(gBuffer[index].pos) + 1.f) * 0.5f * 255.f;

		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, const string displayedData) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);


	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	if (displayedData == "t")
	{
		gbufferTToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
	}
	else if (displayedData == "position")
	{
		gbufferPosToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
	}
	else if (displayedData == "normal")
	{
		gbufferNorToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
	}
	
}

void showImage(uchar4* pbo, int iter, bool isDenoise) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	//sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	if (isDenoise)
	{
		sendDenoisedImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_outDenoisedImage);
	}
	else
	{
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	}
}

void printDenoiseTime()
{
#if DENOISE_TIME
	std::cout << "Denoising Time: " << denoiseTime << std::endl;
#endif
}