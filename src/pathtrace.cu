#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"

#define ERRORCHECK 1
#define SORTMATERIAL 0
#define CACHEINTERSECTION 0
#define THINLENSCAM 0
#define ANTIALIASING 1
#define GBUFFER_NORMAL 0
#define GBUFFER_POSITION 1

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

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendDenoisedImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int idx = x + resolution.x * y;
		glm::vec3 pix = image[idx];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
		
		pbo[idx].w = 0;
		pbo[idx].x = color.x;
		pbo[idx].y = color.y;
		pbo[idx].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_denoisedImage = NULL;
static glm::vec3* dev_denoisedImageSaved = NULL;
static glm::vec3* dev_denoisedIn= NULL;
static Geom* dev_geoms = NULL;
static Triangle* dev_tris = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_intersections_cache = NULL;
static bvhNode* dev_bvhNodes = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static float timer = 0.f;
void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	
	const Camera& cam = hst_scene->state.camera;
	
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_denoisedImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoisedImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_denoisedImageSaved, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoisedImageSaved, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_denoisedIn, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoisedIn, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

	//for BVH tree
	cudaMalloc(&dev_bvhNodes, scene->sceneBVH.nodeCount * sizeof(bvhNode));
	cudaMemcpy(dev_bvhNodes, scene->sceneBVH.bvhNodes.data(), scene->sceneBVH.nodeCount * sizeof(bvhNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_denoisedImage);
	cudaFree(dev_denoisedIn);
	cudaFree(dev_denoisedImageSaved);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_intersections_cache);
	cudaFree(dev_tris);
	cudaFree(dev_gBuffer);
	checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 ConcentricSampleDisk(thrust::default_random_engine& rng) {

	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec2 u = glm::vec2(u01(rng), u01(rng));
	glm::vec2 uOffset = 2.f * u - glm::vec2(1);

	if (uOffset.x == 0 && uOffset.y == 0) {
		return glm::vec2(0.f, 0.f);
	}
	float theta, r;
	if (abs(uOffset.x) > abs(uOffset.y)) {
		r = uOffset.x;
		theta = PiOver4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
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
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		
#if ANTIALIASING
		float dx = 0.f, dy = 0.f;
		thrust::random::uniform_real_distribution<float> u0(-1.f, 1.f);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u0(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u0(rng) - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
#if THINLENSCAM
		if (cam.lensRadius > 0) {
			glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(rng);
			float ft = glm::abs(cam.focalDistance / segment.ray.direction.z);
			glm::vec3 pFocus = getPointOnRay(segment.ray, ft);
			segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0.f);
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);

		}
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
	, Triangle* tri
	, bvhNode* bvh
	, int geoms_size
	, int bvhSize
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
		bool bvhSearched = false;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		
		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];
			int geomID = -1;
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MODEL)
			{
#if BVH
				
				if (!bvhSearched) {
					t = bvhIntersectionTest(geoms, tri, bvh, pathSegment.ray,
						tmp_intersect, tmp_normal, outside, bvhSize, geomID);
					bvhSearched = true;
				}
				
#else
				t = meshIntersectionTest(geom, tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#endif
			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
#if BVH
				hit_geom_index = geom.type == MODEL ? geomID : i;
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
			intersections[path_index].position = intersect_point;
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

__global__ void shadeRealMaterial(
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
		PathSegment& path = pathSegments[idx];
		if (path.remainingBounces == 0) return;
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
				path.color *= (materialColor * material.emittance);
				path.remainingBounces = 0;
			}

			else {
				glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);
				scatterRay(
					path,
					intersectionPoint,
					intersection.surfaceNormal,
					material,
					rng);
			}
		}
		else {
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0;
		}
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
		gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
		gBuffer[idx].position = glm::normalize(shadeableIntersections[idx].position);
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

struct bouncesPred
{
	__host__ __device__
	bool operator()(const PathSegment& p)
	{
		return p.remainingBounces != 0;
	}
};

struct sortMaterialID
{
	__host__ __device__
	bool operator()(const ShadeableIntersection& l, const ShadeableIntersection& r)
	{
		return l.materialId < r.materialId;
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
	if(iter == 1) timer = 0.f;
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

	bool iterationComplete = false;
	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	while (!iterationComplete) {

		
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHEINTERSECTION
		if (iter != 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
#endif
			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_tris
				, dev_bvhNodes
				, hst_scene->geoms.size()
				, hst_scene->sceneBVH.nodeCount
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
#if CACHEINTERSECTION	
		}

	
		if (iter == 1 && depth == 0) {
			cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#endif

		
		//// tracing
		////dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		//computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
		//	depth
		//	, num_paths
		//	, dev_paths
		//	, dev_geoms
		//	, hst_scene->geoms.size()
		//	, dev_intersections
		//	);
		//checkCUDAError("trace one bounce");
		//cudaDeviceSynchronize();
		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}


		depth++;
#if SORTMATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortMaterialID());
#endif


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeRealMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);

		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, bouncesPred());
		num_paths = dev_path_end - dev_paths;
		iterationComplete = num_paths == 0; // TODO: should be based off stream compaction results.
	

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	
#if GBUFFER_NORMAL
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 normal = (gBuffer[index].normal + 1.0f) * 255.0f * 0.5f;
		pbo[index].w = 0;
		pbo[index].x = normal.x;
		pbo[index].y = normal.y;
		pbo[index].z = normal.z;
	}
#elif GBUFFER_POSITION
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 position = gBuffer[index].position * 255.0f * 0.75f;

		pbo[index].w = 0;
		pbo[index].x = position.x;
		pbo[index].y = position.y;
		pbo[index].z = position.z;
	}
#else
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer[index].t * 255.0;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;
	}
#endif
}



__global__ void A_Trous(glm::ivec2 resolution, GBufferPixel* gBuffer,
	float c_phi, float n_phi, float p_phi, float stepwidth, glm::vec3* inputImage, glm::vec3* outputImage) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	glm::vec3 sum = glm::vec3(0.f);
	const float gaussianKernel[5][5] = {    
											{0.0030,    0.0133,    0.0219,    0.0133,    0.0030},
											{0.0133,    0.0596,    0.0983,    0.0596,    0.0133}, 
											{0.0219,    0.0983,    0.1621,    0.0983,    0.0219},
											{0.0133,    0.0596,    0.0983,    0.0596,    0.0133},
											{0.0030,    0.0133,    0.0219,    0.0133,    0.0030} 
										   };


	int curIdx = y * resolution.x + x;
	glm::vec3 cval = inputImage[curIdx];
	glm::vec3 nval = gBuffer[curIdx].normal;
	glm::vec3 pval = gBuffer[curIdx].position;

	float cum_w = 0.f;
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			int nx = x + j * stepwidth;
			int ny = y + i * stepwidth;
			if (nx < 0 || ny < 0 || nx >= resolution.x || ny >= resolution.y) continue;
			
			int nIndex = nx + ny * resolution.x;
			glm::vec3 ctmp = inputImage[nIndex];
			glm::vec3 t = cval - ctmp;

			float dist2 = glm::dot(t, t);
			float c_w = min(exp(-(dist2) / c_phi), 1.f);

			glm::vec3 ntmp = gBuffer[nIndex].normal;
			t = nval - ntmp;
			dist2 = max(glm::dot(t, t) / (stepwidth * stepwidth), 0.f);
			float n_w = min(exp(-(dist2) / n_phi), 1.f);

			glm::vec3 ptmp = gBuffer[nIndex].position;
			t = pval - ptmp;
			dist2 = glm::dot(t, t);
			float p_w = min(exp(-(dist2) / p_phi), 1.f);

			float weight = c_w * n_w * p_w;
			sum += ctmp * weight * gaussianKernel[i+2][j+2];
			cum_w += weight * gaussianKernel[i+2][j+2];
		}
	}
	outputImage[curIdx] = sum / cum_w;
}

//
__global__ void DenoiseInit(int iteration, glm::ivec2 resolution, glm::vec3* image, glm::vec3* denoisedImage)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int idx = x + resolution.x * y;
		denoisedImage[idx].x = image[idx].x / iteration;
		denoisedImage[idx].y = image[idx].y / iteration;
		denoisedImage[idx].z = image[idx].z / iteration;
	}
}

__global__ void DenoiseFinalize(int iteration, glm::ivec2 resolution, glm::vec3* finalOutput, glm::vec3* denoisedImage)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int idx = x + resolution.x * y;
		finalOutput[idx].x = denoisedImage[idx].x  * iteration;
		finalOutput[idx].y = denoisedImage[idx].y  * iteration;
		finalOutput[idx].z = denoisedImage[idx].z  * iteration;
	}
}

void denoise(float cw, float nw, float pw, int filterSize, int iter, int endIteration) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	const int pixelcount = cam.resolution.x * cam.resolution.y;
	DenoiseInit << <blocksPerGrid2d, blockSize2d >> > (iter, cam.resolution, dev_image, dev_denoisedImage);
	//cudaMemcpy(dev_denoisedImage, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaEventRecord(start);
	for (int stepWidth = 1; stepWidth <= filterSize / 2; stepWidth <<= 1){
		A_Trous << < blocksPerGrid2d, blockSize2d >> > (cam.resolution, dev_gBuffer,
			cw, nw, pw, stepWidth, dev_denoisedImage, dev_denoisedIn);
		swap(dev_denoisedImage, dev_denoisedIn);
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Elapsed time at Iteration: " << iter << " is: "<< milliseconds << " ms" << endl;

	timer += milliseconds;
	if (iter == endIteration) cout << "Total denoised time elapsed: " << timer <<" ms"<< endl;
	DenoiseFinalize << <blocksPerGrid2d, blockSize2d >> > (iter, cam.resolution, dev_denoisedImageSaved, dev_denoisedImage);
	//for save
	cudaMemcpy(hst_scene->state.image.data(), dev_denoisedImageSaved, cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter, bool denoise) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	if (denoise) {
		sendDenoisedImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoisedImage);
	}
	else {
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	}
	
}