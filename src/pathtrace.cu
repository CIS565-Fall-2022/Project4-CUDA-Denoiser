#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include "device_launch_parameters.h"


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

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

        //show intersection t
        //float timeToIntersect = gBuffer[index].t * 256.0;
        //pbo[index].w = 0;
        //pbo[index].x = timeToIntersect;
        //pbo[index].y = timeToIntersect;
        //pbo[index].z = timeToIntersect;
        
        //show normal
        // note we need to times 255, if times 256, it becomes 0
        /*pbo[index].w = 0;
        pbo[index].x = glm::abs(gBuffer[index].normal.x) * 255.f;
        pbo[index].y = glm::abs(gBuffer[index].normal.y) * 255.f;
        pbo[index].z = glm::abs(gBuffer[index].normal.z) * 255.f;*/

        //show position
        pbo[index].w = 0;
        pbo[index].x = glm::abs(gBuffer[index].position.x) * 20.f;
        pbo[index].y = glm::abs(gBuffer[index].position.y) * 20.f;
        pbo[index].z = glm::abs(gBuffer[index].position.z) * 20.f;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static float* dev_kernel = NULL;
static glm::ivec2* dev_offset = NULL;
static glm::vec3* dev_image2 = NULL;
static GBufferPixelZ * dev_gBufferZ = NULL;
static float* dev_gaussian = NULL;
static glm::ivec2* dev_gaussianOffset = NULL;

void pathtraceInit(Scene *scene, bool useZforPos, bool gaussian) {
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


    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_image2, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image2, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_kernel, 25 * sizeof(float));
    cudaMalloc(&dev_offset, 25 * sizeof(glm::ivec2));
    setKernelOffset(dev_kernel, dev_offset);
    
    if (useZforPos) {
        cudaMalloc(&dev_gBufferZ, pixelcount * sizeof(GBufferPixelZ));
    }
    else {
        cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
    }

    //just malloc gaussian anyway
    cudaMalloc(&dev_gaussianOffset, 49 * sizeof(glm::ivec2));
    cudaMalloc(&dev_gaussian, 49 * sizeof(float));
    setGaussianOffset(dev_gaussian, dev_gaussianOffset);
    checkCUDAError("pathtraceInit");
}

void pathtraceFree(bool useZforPos, bool gaussian) {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_kernel);
    cudaFree(dev_offset);
    cudaFree(dev_image2);
    if (useZforPos) {
        cudaFree(dev_gBufferZ);
    }
    else {
        cudaFree(dev_gBuffer);
    }
    cudaFree(dev_gaussian);
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
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;  //pixel index x
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;  //pixel index y

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
  GBufferPixel* gBuffer,
    GBufferPixelZ* gBufferZ,
    bool useZforPos
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    if (useZforPos) {
        gBufferZ[idx].normal = shadeableIntersections[idx].surfaceNormal;
        if (shadeableIntersections[idx].t == -1) {
            gBufferZ[idx].z = 0.f;
        }
        else {
            gBufferZ[idx].z = shadeableIntersections[idx].t;
        }
    }
    else {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        if (gBuffer[idx].t == -1.f) {
            gBuffer[idx].position = glm::vec3(0.f);
        }
        else {
            gBuffer[idx].position = pathSegments[idx].ray.origin + gBuffer[idx].t * pathSegments[idx].ray.direction;
        }
    }
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

__host__ __device__ void getPosFromZ(
    int pixelX,
    int pixelY,
    Camera cam,
    float z,
    glm::vec3& pos
) {
    glm::vec3 origin = cam.position;

    glm::vec3 dir = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)pixelX - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)pixelY - (float)cam.resolution.y * 0.5f));
    pos = origin + dir * (z - 0.0000001f);
}

__global__ void kernDenoise(
    int num_paths, 
    glm::vec3* image,
    float* kernel,
    glm::ivec2* offset,
    GBufferPixel* gBuffers,
    int filterSize,
    int num_step,
    Camera cam,
    float c_phi,
    float n_phi,
    float p_phi,
    glm::vec3* image2,
    int iteration,
    GBufferPixelZ* gBufferZ, 
    bool useZforPos,
    int kernSize
) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < num_paths)
    {
        glm::vec3 sum = glm::vec3(0.f);
        glm::vec3 cval = image[index];
        glm::vec3 nval;
        glm::vec3 pval;
        int pixelY = index / cam.resolution.x;
        int pixelX = index - (pixelY * cam.resolution.x);

        if (useZforPos) {
            float z = gBufferZ[index].z;
            getPosFromZ(pixelX, pixelY, cam, z, pval);
            nval = gBufferZ[index].normal;
        }
        else {
            nval = gBuffers[index].normal;
            pval = gBuffers[index].position;
        }
        float cum_w = 0.f;

        //offset: (-2, -2), (-2, -1), (-2, 0), ....
        for (int i = 0; i < kernSize; ++i) {

            int stepSize = pow(2, num_step);
            glm::ivec2 currOffset = offset[i] * stepSize;

            //keep currPixel inside image
            int currPixelX = glm::clamp(currOffset.x + pixelX, 0, cam.resolution.x - 1);
            int currPixelY = glm::clamp(currOffset.y + pixelY, 0, cam.resolution.y - 1);
            int currIndex = currPixelX + (currPixelY * cam.resolution.x);    

            glm::vec3 ctmp = image[currIndex];
            glm::vec3 t = cval - ctmp;
            
            //increase color weight when iteration is higher
            float dist2 = glm::dot(t, t);
            float c_w = glm::min(glm::exp(-(dist2) / c_phi), 1.f);

            glm::vec3 ptmp;
            glm::vec3 ntmp;
            if (useZforPos) {
                ntmp = gBufferZ[currIndex].normal;
            }
            else {
                ntmp = gBuffers[currIndex].normal;
            }
            t = nval - ntmp;
            dist2 = glm::max(glm::dot(t, t) / (stepSize * stepSize), 0.f);
            float n_w = glm::min(glm::exp(-(dist2) / n_phi), 1.f);

            if (useZforPos) {
                float currZ = gBufferZ[currIndex].z;
                getPosFromZ(currPixelX, currPixelY, cam, currZ, ptmp);
            }
            else {
                ptmp = gBuffers[currIndex].position;
            }
            t = pval - ptmp;
            dist2 = glm::dot(t, t);
            float p_w = glm::min(glm::exp(-(dist2) / p_phi), 1.f);

            float weight = c_w * n_w * p_w;

            sum += ctmp * weight * kernel[i];
            cum_w += weight * kernel[i];
        }
        image2[index] = sum / cum_w;
    }
}

struct remainingBounceIsNot0 {
    __host__ __device__
        bool operator()(const PathSegment& p1) {
        return (p1.remainingBounces > 0);
    }
};

__global__ void kernImageCopy(
    int num_paths,
    glm::vec3* image1,
    glm::vec3* image2
) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < num_paths) {
        image1[index] = image2[index];
    }
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, int filterSize, float c_phi, float n_phi, float p_phi, bool denoiser, bool useZforPos, bool gaussian) {
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
    int curr_num_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
    if (denoiser) {
        // Empty gbuffer
        cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
    }
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

        if (denoiser) {
            if (depth == 0) {
                generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer, dev_gBufferZ, useZforPos);
            }
        }

	    ++depth;

        shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + curr_num_paths, remainingBounceIsNot0());
        curr_num_paths = dev_path_end - dev_paths;

        iterationComplete = ((depth == traceDepth) || (curr_num_paths == 0));
	}

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    if (denoiser) {
        //filter size represents one dimension of the filter
        int num_steps = ceil(log2(filterSize/2));
        if (num_steps != 0) {
            for (int i = 0; i < num_steps; ++i) {
                if (gaussian) {
                    int kernSize = 49;
                    kernDenoise << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_gaussian, dev_gaussianOffset, dev_gBuffer, filterSize, i, cam, c_phi, n_phi, p_phi, dev_image2, iter, dev_gBufferZ, useZforPos, kernSize);
                }
                else {
                    int kernSize = 25;
                    kernDenoise << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_kernel, dev_offset, dev_gBuffer, filterSize, i, cam, c_phi, n_phi, p_phi, dev_image2, iter, dev_gBufferZ, useZforPos, kernSize);
                }
                if (i != (num_steps - 1)) {
                    cudaMemcpy(dev_image, dev_image2, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
                }
            }
        }
        cudaMemcpy(hst_scene->state.image.data(), dev_image2,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU

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

void setKernelOffset(float* dev_kernel, glm::ivec2* dev_offset) {
    //set offset
    //(-2, -2), (-2, -1), (-2, 0), ....
    int offsetCount = 0;
    for (int i = -2; i < 3; ++i) {
        for (int j = -2; j < 3; ++j) {
            cudaMemcpy(dev_offset + offsetCount, &(glm::ivec2(i, j)), sizeof(glm::ivec2), cudaMemcpyHostToDevice);
            ++offsetCount;
        }
    }

    std::vector<float> kernelNominator = { 1.f, 4.f, 7.f, 4.f, 1.f,
                                          4.f, 16.f, 26.f, 16.f, 4.f,
                                          7.f, 26.f, 41.f, 26.f, 7.f,
                                          4.f, 16.f, 26.f, 16.f, 4.f,
                                          1.f, 4.f, 7.f, 4.f, 1.f };
    for (int i = 0; i < 25; ++i) {
        kernelNominator[i] /= 273.f;
        cudaMemcpy(dev_kernel + i, &kernelNominator[i], sizeof(float), cudaMemcpyHostToDevice);
    }
}

void setGaussianOffset(float* gaussian, glm::ivec2* gaussianOffset) {
    
    std::vector<float> kernelNominator = { 0.f, 0.f, 1.f, 2.f, 1.f, 0.f, 0.f,
                                          0.f, 3.f, 13.f, 22.f, 13.f, 3.f, 0.f,
                                          1.f, 13.f, 59.f, 97.f, 59.f, 13.f, 1.f,
                                          2.f, 22.f, 97.f, 159.f, 97.f, 22.f, 2.f,
                                          1.f, 13.f, 59.f, 97.f, 59.f, 13.f, 1.f, 
                                          0.f, 3.f, 13.f, 22.f, 13.f, 3.f, 0.f,
                                          0.f, 0.f, 1.f, 2.f, 1.f, 0.f, 0.f
    };
    for (int i = 0; i < 49; ++i) {
        kernelNominator[i] /= 1003.f;
        cudaMemcpy(gaussian + i, &kernelNominator[i], sizeof(float), cudaMemcpyHostToDevice);
    }

    int offsetCount = 0;
    for (int i = -3; i < 4; ++i) {
        for (int j = -3; j < 4; ++j) {
            cudaMemcpy(gaussianOffset + offsetCount, &(glm::ivec2(i, j)), sizeof(glm::ivec2), cudaMemcpyHostToDevice);
            ++offsetCount;
        }
    }
}