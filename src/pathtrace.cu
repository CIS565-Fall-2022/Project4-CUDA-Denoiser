#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

//#define DISPLAY_POSITION
//#define DISPLAY_NORMAL
//#define DISPLAY_DEPTH

#define WAVELET

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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        //float timeToIntersect = gBuffer[index].t * 256.0;

        pbo[index].w = 0;
#ifdef DISPLAY_NORMAL
        pbo[index].x = ((gBuffer[index].nor.x + 1.0f) * 0.5f) * 255.0f;
        pbo[index].y = ((gBuffer[index].nor.y + 1.0f) * 0.5f) * 255.0f;
        pbo[index].z = ((gBuffer[index].nor.z + 1.0f) * 0.5f) * 255.0f;
#endif

#ifdef DISPLAY_POSITION
        pbo[index].x = glm::abs(gBuffer[index].pos.x) * 0.1f * 255.0f;
        pbo[index].y = glm::abs(gBuffer[index].pos.y) * 0.1f * 255.0f;
        pbo[index].z = glm::abs(gBuffer[index].pos.z) * 0.1f * 255.0f;
#endif

#ifdef DISPLAY_DEPTH
        pbo[index].x = glm::abs(gBuffer[index].t) * 0.1f * 255.0f;
        pbo[index].y = glm::abs(gBuffer[index].t) * 0.1f * 255.0f;
        pbo[index].z = glm::abs(gBuffer[index].t) * 0.1f * 255.0f;
#endif
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static glm::vec3* dev_image_denoised_a = NULL;
static glm::vec3* dev_image_denoised_b = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...


const float h_kernel[25] = {    1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f,
                                1.0 / 64.0f, 1.0f / 16.0f, 3.0 / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
                                3.0f / 128.0f, 3.0f / 32.0f, 9.0f / 64.0f, 3.0f / 32.0f, 3.0f / 128.0f,
                                1.0 / 64.0f, 1.0f / 16.0f, 3.0 / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
                                1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f};

static float* dev_h_kernel = NULL;

const glm::ivec2 h_kernel_offsets[25] = {   glm::ivec2(-2,-2), glm::ivec2(-1,-2), glm::ivec2(0,-2), glm::ivec2(1,-2), glm::ivec2(2,-2),
                                            glm::ivec2(-2,-1), glm::ivec2(-1,-1), glm::ivec2(0,-1), glm::ivec2(1,-1), glm::ivec2(2,-1),
                                            glm::ivec2(-2,0), glm::ivec2(-1,0), glm::ivec2(0,0), glm::ivec2(1,0), glm::ivec2(2,0),
                                            glm::ivec2(-2,1), glm::ivec2(-1,1), glm::ivec2(0,1), glm::ivec2(1,1), glm::ivec2(2,1),
                                            glm::ivec2(-2,2), glm::ivec2(-1,2), glm::ivec2(0,2), glm::ivec2(1,2), glm::ivec2(2,2) };

static glm::ivec2* dev_h_kernel_offsets = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_image_denoised_a, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoised_a, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_image_denoised_b, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoised_b, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_h_kernel, 25 * sizeof(float));
    cudaMemcpy(dev_h_kernel, h_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_h_kernel_offsets, 25 * sizeof(glm::ivec2));
    cudaMemcpy(dev_h_kernel_offsets, h_kernel_offsets, 25 * sizeof(glm::ivec2), cudaMemcpyHostToDevice);

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_image_denoised_a);
    cudaFree(dev_image_denoised_b);
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    cudaFree(dev_h_kernel);
    cudaFree(dev_h_kernel_offsets);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

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
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces == 0) {
      return;
    }

    if (intersection.t > 0.0f) { // if the intersection exists...
      segment.remainingBounces--;
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        segment.color *= (materialColor * material.emittance);
        segment.remainingBounces = 0;
      }
      else {
        segment.color *= materialColor;
        glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
        scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      segment.color = glm::vec3(0.0f);
      segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
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
    gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].pos = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * shadeableIntersections[idx].t;
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Pathtracing Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * NEW: For the first depth, generate geometry buffers (gbuffers)
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally:
    //     * if not denoising, add this iteration's results to the image
    //     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
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

	depth++;

  shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  iterationComplete = depth == traceDepth;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

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
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
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

#ifdef WAVELET

// implements the a trous (ratatouille) filter
__global__ void denoiseRatatouille(glm::ivec2 resolution, int iter, float* h_kernel, glm::ivec2* h_kernel_offsets, int iter_offset, float sigma_p_sq, float sigma_n_sq, float sigma_rt_sq, GBufferPixel* gBuffer, glm::vec3* image, glm::vec3* image_denoised) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        glm::vec3 color_accum = glm::vec3(0, 0, 0);
        int x_image;
        int y_image;
        int kernel_index;
        glm::ivec2 kernel_offset;
        float kernel_val;
        
        glm::vec3 input_color = image[x + (y * resolution.x)];
        GBufferPixel input_gBuffer = gBuffer[x + (y * resolution.x)];
        glm::vec3 buffer_diff;
        float squared_distance;
        float w_rt;
        float w_n;
        float w_p;
        float total_weight;
        float w_accum = 0.0f;


        for (int i = 0; i < 25; ++i) {
            kernel_offset = h_kernel_offsets[i] * iter_offset;
            kernel_val = h_kernel[i];

            x_image = glm::clamp(x + kernel_offset.x, 0, resolution.x - 1);
            y_image = glm::clamp(y + kernel_offset.y, 0, resolution.y - 1);

            buffer_diff = input_gBuffer.pos - gBuffer[x_image + (y_image * resolution.x)].pos;
            squared_distance = glm::dot(buffer_diff, buffer_diff);
            w_p = glm::min(glm::exp(-(squared_distance) / sigma_p_sq), 1.0f);

            buffer_diff = input_gBuffer.nor - gBuffer[x_image + (y_image * resolution.x)].nor;
            squared_distance = glm::max(glm::dot(buffer_diff, buffer_diff) / ((float)(iter_offset * iter_offset)), 0.0f);
            w_n = glm::min(glm::exp(-(squared_distance) / sigma_n_sq), 1.0f);

            buffer_diff = input_color - image[x_image + (y_image * resolution.x)];
            squared_distance = glm::dot(buffer_diff, buffer_diff);
            w_rt = glm::min(glm::exp(-(squared_distance) / sigma_rt_sq), 1.0f);

            

            total_weight = w_p * w_n * w_rt;

            color_accum += total_weight * kernel_val * image[x_image + (y_image * resolution.x)];
            w_accum += total_weight * kernel_val;
        }
        
        image_denoised[x + (y * resolution.x)] = color_accum / w_accum;
    }
}


void denoiseAndShowImage(uchar4* pbo, int iter, DenoiseParams denoise_params) {
    if (denoise_params.denoise) {
        const Camera& cam = hst_scene->state.camera;
        const dim3 blockSize2d(8, 8);
        const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

        PerformanceTimer perf_timer;
        perf_timer.startGpuTimer();
        cudaMemcpy(dev_image_denoised_a, dev_image, cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        
        int iter_offset = 1;
        
        for (int i = 0; i < denoise_params.denoise_iterations; ++i) {
            denoiseRatatouille << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_h_kernel, dev_h_kernel_offsets, iter_offset, denoise_params.sigma_p * denoise_params.sigma_p, denoise_params.sigma_n * denoise_params.sigma_n, denoise_params.sigma_rt * denoise_params.sigma_rt, dev_gBuffer, dev_image_denoised_a, dev_image_denoised_b);
            if (i < denoise_params.denoise_iterations - 1) {
                glm::vec3* temp = dev_image_denoised_b;
                dev_image_denoised_b = dev_image_denoised_a;
                dev_image_denoised_a = temp;
            }
            iter_offset *= 2;
            denoise_params.sigma_rt /= (float)iter_offset;
        }


        // Send results to OpenGL buffer for rendering
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image_denoised_b);
        perf_timer.endGpuTimer();
        std::cout << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
    }
    
}

#else
// implements the a trous (ratatouille) filter
__global__ void denoiseRatatouille(glm::ivec2 resolution, int iter, float* h_kernel, glm::ivec2* h_kernel_offsets, int iter_offset, GBufferPixel* gBuffer, glm::vec3* image, glm::vec3* image_denoised) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        glm::vec3 color_accum = glm::vec3(0, 0, 0);
        int x_image;
        int y_image;
        int kernel_index;
        glm::ivec2 kernel_offset;
#pragma unroll
        for (int i = 0; i < 25; ++i) {
            kernel_offset = h_kernel_offsets[i] * iter_offset;

            x_image = glm::clamp(x + kernel_offset.x, 0, resolution.x - 1);
            y_image = glm::clamp(y + kernel_offset.y, 0, resolution.y - 1);
            color_accum += h_kernel[i] * image[x_image + (y_image * resolution.x)];
        }

        image_denoised[x + (y * resolution.x)] = color_accum;
    }
}

void denoiseAndShowImage(uchar4* pbo, int iter, DenoiseParams denoise_params) {
    if (denoise_params.denoise) {
        const Camera& cam = hst_scene->state.camera;
        const dim3 blockSize2d(8, 8);
        const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

        PerformanceTimer perf_timer;
        perf_timer.startGpuTimer();
        cudaMemcpy(dev_image_denoised_a, dev_image, cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

        int iter_offset = 1;
        for (int i = 0; i < denoise_params.denoise_iterations; ++i) {
            denoiseRatatouille << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_h_kernel, dev_h_kernel_offsets, iter_offset, dev_gBuffer, dev_image_denoised_a, dev_image_denoised_b);
            if (i < denoise_params.denoise_iterations - 1) {
                glm::vec3* temp = dev_image_denoised_b;
                dev_image_denoised_b = dev_image_denoised_a;
                dev_image_denoised_a = temp;
            }
            iter_offset *= 2;

        }


        // Send results to OpenGL buffer for rendering
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image_denoised_b);
        perf_timer.endGpuTimer();
        std::cout << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
    }

}
#endif
