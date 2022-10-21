#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define SORTMATERIALS 0
#define CACHE 0
#define ANTIALIAS 0

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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;
        glm::vec3 normal = gBuffer[index].normal;
        glm::ivec3 normalColor;
        normalColor.x = glm::clamp((int)abs((normal.x * 255.0)), 0, 255);
        normalColor.y = glm::clamp((int)abs((normal.y * 255.0)), 0, 255);
        normalColor.z = glm::clamp((int)abs((normal.z * 255.0)), 0, 255);
        pbo[index].w = 0;
        //pbo[index].x = timeToIntersect;
        //pbo[index].y = timeToIntersect;
        //pbo[index].z = timeToIntersect;
        float x = gBuffer[index].normal.x;
        float y = gBuffer[index].normal.y;
        float z = gBuffer[index].normal.z;
        pbo[index].x = normalColor.x;
        pbo[index].y = normalColor.y;
        pbo[index].z = normalColor.z;
        //glm::vec3 position = gBuffer[index].position;
        //glm::ivec3 positionColor;
        //positionColor.x = glm::clamp((int)abs((position.x * 32.0)), 0, 255);
        //positionColor.y = glm::clamp((int)abs((position.y * 32.0)), 0, 255);
        //positionColor.z = glm::clamp((int)abs((position.z * 32.0)), 0, 255);
        //pbo[index].x = positionColor.x;
        //pbo[index].y = positionColor.y;
        //pbo[index].z = positionColor.z;

    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static float host_kernel[25] = { 1 / 256 ,  1 / 64 , 3 / 128 , 1 / 64 , 1 / 256,
    1 / 64 , 1 / 16 , 3 / 32 , 1 / 16 , 1 / 64,
    3 / 128 , 3 / 32 , 9 / 64 , 3 / 32 , 3 / 128,
    1 / 64 , 1 / 16 , 3 / 32 , 1 / 16 , 1 / 64,
    1 / 256 , 1 / 64 , 3 / 128 , 1 / 64 , 1 / 256 };
static float* dev_kernel = NULL;
static glm::vec3* dev_image_denoise = NULL;
static glm::vec3* dev_image_denoise_ping_pong = NULL;
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_image_buffer = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...
#if CACHE
static bool hasCache = false;
static ShadeableIntersection* dev_intersectionsFirstCache = NULL;
#endif

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

	cudaMalloc(&dev_image_buffer, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_buffer, 0, pixelcount * sizeof(glm::vec3));
	
	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
	// TODO: initialize any extra device memeory you need
#if CACHE
	cudaMalloc(&dev_intersectionsFirstCache, pixelcount * sizeof(ShadeableIntersection));

#endif

	checkCUDAError("pathtraceInit");
    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_kernel, 25 * sizeof(float));
    cudaMemcpy(dev_kernel, host_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_image_denoise, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoise, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_image_denoise_ping_pong, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoise_ping_pong, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceInit");
}
void pathtraceInitCheckpoint(Scene* scene)
{
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMemcpy(dev_image, hst_scene->state.image.data(),
		pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAError("pathtraceCheckpointInit");

}


void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_image_buffer);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if CACHE
	cudaFree(dev_intersectionsFirstCache);
#endif 
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    cudaFree(dev_kernel);
    cudaFree(dev_image_denoise);
    cudaFree(dev_image_denoise_ping_pong);
    // TODO: clean up any extra device memory you created

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

__device__ __host__ glm::vec2 concentricDisc(glm::vec2 u, thrust::default_random_engine rng)
{
	thrust::uniform_real_distribution<float> uNeg11(-1, 1);
	glm::vec2 offset(uNeg11(rng), uNeg11(rng));
	float theta;
	float r;
	if (offset.x == 0.0f && offset.y == 0.0f)
	{
		return glm::vec2(0, 0);
	}
	if (std::abs(offset.x) > std::abs(offset.y))
	{
		r = offset.x;
		theta = PI / 4.0f * (offset.y / offset.x);

	}
	else
	{
		r = offset.y;
		theta = PI / 2.0f - PI / 4.0f * (offset.x / offset.y);
	}
	return r * glm::vec2(cos(theta), sin(theta));
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	float lens_radius = cam.lens_radius; 
	float focal_length = cam.focal_length;
	bool dof = cam.dof;
	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::uniform_real_distribution<float> uNeg11(-1, 1);
		auto rng = makeSeededRandomEngine(iter, index, traceDepth);
;
		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// implemented antialiasing by jittering the ray
#if ANTIALIAS && !CACHE
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y  - (float)cam.resolution.y * 0.5f)
		);
#endif
#if !CACHE
		if (dof)
		{
			auto lens = makeSeededRandomEngine(iter, index, focal_length);
			glm::vec2 offset = concentricDisc(glm::vec2(), lens) * lens_radius;
			glm::vec3 focal_point = getPointOnRay(segment.ray, focal_length);
			segment.ray.origin.x += offset.x;
			segment.ray.origin.y += offset.y;
			segment.ray.direction = glm::normalize(focal_point - segment.ray.origin);
		}

#endif


		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
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

		if (hit_geom_index == -1 )
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].outside = outside;

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
		if (intersection.t > 0.0f ) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not
				//glm::vec3 intersection_point = pathSegments[idx].ray.origin + (intersection.t * pathSegments[idx].ray.direction);
				glm::vec3 intersection_point = getPointOnRay(pathSegments[idx].ray, intersection.t);
				scatterRay(pathSegments[idx], intersection_point, intersection.surfaceNormal, material, rng, intersection.outside);
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

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, glm::vec3* image_buffer, PathSegment* iterationPaths, int iter)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}
__global__ void denoise(GBufferPixel* gBuffer, Camera cam, glm::vec3* image, float* dumb_kernel, glm::vec3* image_denoise, int level, float colorWeight, float normalWeight, float positionWeight)
{
    //TODO: Test fastest way
    float kernel_25[25] = { 1.0f / 256.0f ,  1.0f / 64.0f , 3.0f / 128.0f , 1.0f / 64.0f , 1.0f / 256.0f,
    1.0f / 64.0f , 1.0f / 16.0f , 3.0f / 32.0f , 1.0f / 16.0f , 1.0f / 64.0f,
    3.0f / 128.0f , 3.0f / 32.0f , 9.0f / 64.0f , 3.0f / 32.0f , 3.0f / 128.0f,
    1.0f / 64.0f , 1.0f / 16.0f , 3.0f / 32.0f , 1.0f / 16.0f , 1.0f / 64.0f,
    1.0f / 256.0f , 1.0f / 64.0f , 3.0f / 128.0f , 1 / 64.0f , 1.0f / 256.0f };
    float kernel[5] = { 1.f / 16.f, 1.f / 4.f, 3.f / 8.f, 1.f / 4.f , 1.f / 16.f };
    int x_o = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_o = (blockIdx.y * blockDim.y) + threadIdx.y;
    int origin_index = x_o + (y_o * cam.resolution.x);
    float cum_weight = 0;
    glm::vec3 sum;
    if (x_o < cam.resolution.x && y_o < cam.resolution.y)
    {
#pragma unroll
        for (int j = 0; j < 5; j++)
        {
#pragma unroll
            for (int i = 0; i < 5; i++)
            {
                int x = x_o + (i - 2) * (1 << level);
                int y = y_o + (j - 2) * (1 << level);
                // x = glm::clamp(x, 0, cam.resolution.x - 1);
                // y = glm::clamp(y, 0, cam.resolution.y - 1);
                if (x < cam.resolution.x && y < cam.resolution.y && (x >= 0) && (y) >= 0)
                {
                    //Cap weights at 1.0 per paper
                    int index = (x + (y * cam.resolution.x));
                    float kernel_weight = kernel[i] * kernel[j];
                    glm::vec3 t = image[origin_index] - image_denoise[index];
                    float c_w = glm::min(glm::exp(-glm::dot(t, t) / (colorWeight)),1.0f);
                    t = gBuffer[origin_index].normal - gBuffer[index].normal;
                    //Need to update based on step size?
                    float n_w = glm::min(glm::exp(-glm::dot(t, t) / (normalWeight )), 1.0f);
                    t = gBuffer[origin_index].position - gBuffer[index].position;
                    float p_w = glm::min(glm::exp(-glm::dot(t, t) / (positionWeight)), 1.0f);
                    float weight = c_w * n_w * p_w;
                    sum += (kernel_weight * image_denoise[index] * weight);
                    cum_weight += kernel_weight * weight;
                }
            }

        }

        image[origin_index] = sum/cum_weight;
    }
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
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

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	bool iterationComplete = false;
	int running_paths = num_paths;
	while (!iterationComplete) {
		//std::cout << running_paths << std::endl;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (running_paths + blockSize1d - 1) / blockSize1d;


#if CACHE
		if (hasCache && depth == 0)
		{
			cudaMemcpy(dev_intersections, dev_intersectionsFirstCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, running_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
		}
		if (!hasCache && depth == 0)
		{
			cudaMemcpy(dev_intersectionsFirstCache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, running_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;
		
#if SORTMATERIALS		//IF SORTMATERIALS
		sortMaterials(dev_intersections, dev_paths, running_paths);
#endif
  bool iterationComplete = false;
	while (!iterationComplete) {

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();

  if (depth == 0) {
    generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
  }

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			running_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);

		running_paths = streamCompact(dev_intersections, dev_paths, running_paths);
		if (running_paths == 0 )
		{
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
		

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_image_buffer, dev_paths, iter);

    ///////////////////////////////////////////////////////////////////////////
    //if (iter == 10)
    //{
    //    cudaMemcpy(dev_image_denoise, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    //    for (int sweep = 0; sweep < 4; sweep++)
    //    {
    //        std::swap(dev_image, dev_image_denoise);
    //        //cudaMemcpy(dev_image_denoise, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    //        denoise << < blocksPerGrid2d, blockSize2d >> > (cam, num_paths, dev_image, dev_paths, dev_kernel, dev_image_denoise, sweep);
    //        cudaDeviceSynchronize();    
    //    }
    //    
    //}
    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
void runDenoiser(int filterSize, float colorWeight, float normalWeight, float positionWeight)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    //Avoid negative values
    int denoise_interations = filterSize < 5 ? 0 : log2((filterSize / 5.0f));
    std::cout << denoise_interations << std::endl;;
    cudaMemcpy(dev_image_denoise, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_image_denoise_ping_pong, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    int num_paths = 0;
    for (int sweep = 0; sweep < denoise_interations; sweep++)
    {
        std::swap(dev_image_denoise, dev_image_denoise_ping_pong);
        denoise << < blocksPerGrid2d, blockSize2d >> > (dev_gBuffer, cam, dev_image_denoise, dev_kernel, dev_image_denoise_ping_pong, sweep, colorWeight * colorWeight, normalWeight * normalWeight, positionWeight * positionWeight);
        cudaDeviceSynchronize();
    }
    checkCUDAError("denoise");

}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
}



void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
void showDenoisedImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image_denoise);
}
	checkCUDAError("pathtrace");
	//std::cout << "Running Complete: " << running_paths << std::endl;
	//system("pause");
}
struct is_not_terminated
{
	__host__ __device__ bool operator() (const ShadeableIntersection& x)
	{
		return(x.t != -1.0f);
	}

};

struct bounces_not_terminated
{
	__host__ __device__ bool operator() (const PathSegment& x)
	{
		return(x.remainingBounces > 0);
	}

};
int streamCompact(ShadeableIntersection* intersections, PathSegment* paths, int num_paths)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths(paths);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(intersections);

	//Stencil
	auto new_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, dev_thrust_intersections, is_not_terminated());
	//No Stencil
	int num = (new_end - dev_thrust_paths);
	new_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num, bounces_not_terminated());
	num = (new_end - dev_thrust_paths);
	return num;
}

struct material_id_greater
{
	__host__ __device__ bool operator() (const ShadeableIntersection& x, const ShadeableIntersection& y)
	{
		return(x.materialId > y.materialId);
	}

};

void sortMaterials(ShadeableIntersection* intersections, PathSegment* paths, int num_paths)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths(paths);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(intersections);

	thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths, material_id_greater());
}