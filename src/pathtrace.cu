#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "main.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define DENOISE 1

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

__global__ void gbufferToPBO(uchar4* pbo, glm::vec3* image, glm::ivec2 resolution, GBufferPixel* gBuffer, int type) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    float timeToIntersect = gBuffer[index].t * 256.0;

    glm::vec3 output;
    if (type == 0) {
      output.x = glm::abs(gBuffer[index].normal.x * 255.f);
      output.y = glm::abs(gBuffer[index].normal.y * 255.f);
      output.z = glm::abs(gBuffer[index].normal.z * 255.f);
    }
    else if (type == 1) {
      output = gBuffer[index].pos * 256.f * 0.1f;
    }
    else {
      output = glm::vec3(gBuffer[index].t) * 256.f;
    }

    pbo[index].w = 0;
    pbo[index].x = output.x;
    pbo[index].y = output.y;
    pbo[index].z = output.z;

    image[index].x = output.x;
    image[index].y = output.y;
    image[index].z = output.z;
  }
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

glm::vec3* dev_image_denoise = NULL;
glm::vec3* dev_image_denoise_tmp = NULL;
glm::vec3* dev_image_gBuffer = NULL;

void pathtraceInit(Scene* scene) {
  hst_scene = scene;
  const Camera& cam = hst_scene->state.camera;
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

  cudaMalloc(&dev_image_denoise, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image_denoise, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_image_denoise_tmp, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image_denoise_tmp, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_image_gBuffer, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image_gBuffer, 0, pixelcount * sizeof(glm::vec3));

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

  cudaFree(dev_image_denoise);
  cudaFree(dev_image_denoise_tmp);
  cudaFree(dev_image_gBuffer);

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

__global__ void shadeSimpleMaterials(
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
    }
    else {
      segment.color = glm::vec3(0.0f);
      segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
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

__global__ void denoiseATour(const Camera cam, const int stepWidth, const float c_phi, const float p_phi, const float n_phi,
  const glm::vec3* image, glm::vec3* imageDenoise, const GBufferPixel* gBuffer)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {

    const float gaussian[9] = { 0.00390625, 0.015625, 0.0234375, 0.015625, 0.0625, 0.09375, 0.0234375, 0.09375, 0.140625 };

    glm::vec3 sum = glm::vec3(0.f);
    float cum_w = 0.f;

    int index = x + (y * cam.resolution.x);

    glm::vec3 cval = image[index];
    glm::vec3 pval = gBuffer[index].pos;
    glm::vec3 nval = gBuffer[index].normal;

    for (int j = -2; j <= 2; j++) {
      for (int i = -2; i <= 2; i++) {

        int uvX = min(max(int(x + i * stepWidth), 0), cam.resolution.x - 1);
        int uvY = min(max(int(y + j * stepWidth), 0), cam.resolution.y - 1);

        if (uvX < 0 || uvX >= cam.resolution.x) continue;
        if (uvY < 0 || uvY >= cam.resolution.y) continue;

        float kernelValue = gaussian[-abs(i) + 2 + (-abs(j) + 2) * 3];

        int itmp = uvX + cam.resolution.x * uvY;

        glm::vec3 ctmp = image[itmp];
        glm::vec3 t = cval - ctmp;
        float dist2 = glm::dot(t, t);
        float c_w = glm::min(std::exp(-(dist2) / (c_phi + EPSILON)), 1.f);

        t = nval - gBuffer[itmp].normal;
        dist2 = glm::max(glm::dot(t, t) / (stepWidth * stepWidth), 0.f);
        float n_w = glm::min(std::exp(-(dist2) / (n_phi + EPSILON)), 1.f);

        t = pval - gBuffer[itmp].pos;
        dist2 = glm::dot(t, t);
        float p_w = glm::min(std::exp(-(dist2) / (p_phi + EPSILON)), 1.f);

        float weight = c_w * p_w * n_w;
        sum += ctmp * weight * kernelValue;
        cum_w += weight * kernelValue;
      }
    }

    imageDenoise[index] = sum / cum_w;
  }
}

__global__ void diffuseImage(int nPaths, int iter, glm::vec3* image) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    glm::vec3 color = image[index];
    color.r /= iter;
    color.b /= iter;
    color.g /= iter;

    image[index] = color;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool isLast) {
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
  while (!iterationComplete) {

    // tracing
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
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
      generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
    }

    depth++;

    shadeSimpleMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
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
  finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

  ///////////////////////////////////////////////////////////////////////////

#if DENOISE

  if (isLast) {
    cudaMemcpy(dev_image_denoise_tmp, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

    diffuseImage << <numBlocksPixels, blockSize1d >> > (pixelcount, iter, dev_image_denoise_tmp);

    for (int stepWidth = 1; stepWidth * 4 <= ui_filterSize; stepWidth <<= 1) {
      denoiseATour << <blocksPerGrid2d, blockSize2d >> > (cam, stepWidth, ui_colorWeight, ui_positionWeight, ui_normalWeight,
        dev_image_denoise_tmp, dev_image_denoise, dev_gBuffer);

      std::swap(dev_image_denoise_tmp, dev_image_denoise);
    }
    std::swap(dev_image_denoise_tmp, dev_image_denoise);
  }
#endif

  // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
  // Otherwise, screenshots are also acceptable.
  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
    pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, int iter, int type) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
  gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, dev_image_gBuffer, cam.resolution, dev_gBuffer, type);

  cudaMemcpy(hst_scene->state.image.data(), dev_image_gBuffer,
    cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  for (int x = 0; x < cam.resolution.x; x++) {
    for (int y = 0; y < cam.resolution.y; y++) {
      int index = x + (y * cam.resolution.x);
      hst_scene->state.image[index] *= iter;
    }
  }
}

void showImage(uchar4* pbo, int iter) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void showImageDenoise(uchar4* pbo, int iter) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, 1, dev_image_denoise);

  cudaMemcpy(hst_scene->state.image.data(), dev_image_denoise,
    cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  for (int x = 0; x < cam.resolution.x; x++) {
    for (int y = 0; y < cam.resolution.y; y++) {
      int index = x + (y * cam.resolution.x);
      hst_scene->state.image[index] *= iter;
    }
  }
}