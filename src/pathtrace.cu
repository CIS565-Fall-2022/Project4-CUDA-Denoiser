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

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

//Personal Trigger(use to do performance analysis)

#define ENABLE_FIRST_INTERSECTION_CACHE 1

#define ENABLE_RAY_SORTING 1

#define USE_GBUFFER_METHOD 1

#define USE_DEPTH_RECONSTRUCT 0
#define NOT_USE_DEPTH_RECONSTRUCT 1


//Add timer to do data analysis

#define Timer 1

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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, 
	GBufferPixel* gBuffer,GBufferMode mode,const Camera cam) 
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		switch (mode)
		{
		    case Time:
			   float timeToIntersect = gBuffer[index].t * 255.0f;
			   pbo[index].w = 0;
			   pbo[index].x = timeToIntersect;
			   pbo[index].y = timeToIntersect;
			   pbo[index].z = timeToIntersect;
			break;
			case Position:
				glm::vec3 pos = 0.1f * gBuffer[index].position * 255.0f;
				pbo[index].w = 0;
				pbo[index].x = abs(pos.x);
				pbo[index].y = abs(pos.y);
				pbo[index].z = abs(pos.z);
			break;
			case Normal:
				glm::vec3 normal = gBuffer[index].normal;
				pbo[index].w = 0;
				//remap to color(255)
				pbo[index].x = abs((int)(normal.x * 255.0));
				pbo[index].y = abs((int)(normal.y * 255.0));
				pbo[index].z = abs((int)(normal.z * 255.0));
		    break;
			case Depth:
				float z = gBuffer[index].depth;
				z = pow(z, 14.f);
				z*= 255.0f;

				pbo[index].w = 0.0;
				pbo[index].x = z;
				pbo[index].y = z;
				pbo[index].z = z;
				break;
		}
	}
}




static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;

static Geom* dev_geoms = NULL;

static Material* dev_materials = NULL;

static Triangle* dev_triangles = NULL;

static PathSegment* dev_paths = NULL;
//Added
static PathSegment* dev_final_paths = NULL;

static ShadeableIntersection* dev_intersections = NULL;

static GBufferPixel* dev_gBuffer = NULL;
static glm::vec3* dev_denoised_img = NULL;


// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> thrust_dev_paths;
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections;
static int* dev_materialID = NULL;
static thrust::device_ptr<int> thrust_dev_materialID;
static int* dev_materialID_copy = NULL;
static thrust::device_ptr<int> thrust_dev_materialIDCpy;

//Texture Data
static cudaTextureObject_t* dev_texObjs=NULL;
static std::vector<cudaArray_t> dev_texArray;
static std::vector<cudaTextureObject_t> texObjs;

//Mesh data for GPU
static PrimitiveData dev_prim_data;

static Mesh* dev_meshes = NULL;

#if ENABLE_FIRST_INTERSECTION_CACHE
static ShadeableIntersection* dev_first_intersect;
bool bounceAlreadyCached = false;
#endif

#if Timer
static cudaEvent_t startEvent = NULL;
static cudaEvent_t endEvent = NULL;
#endif

void textureInit(const Texture& tex,int i)
{
	//Allocate CUDA Array
	//From NVIDIA Document
	cudaTextureObject_t texObj;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	//allocate texture memory
	cudaMallocArray(&dev_texArray[i],&channelDesc,tex.width,tex.height);

	//Set pitch of sourece (the width in memory in bytes of the 2D array pointed to src)
	//don't have padding const size_t
    // Copy texture image in host memory to device memory
	cudaMemcpyToArray(dev_texArray[i],0,0,tex.image,tex.width*tex.height*tex.component*sizeof(unsigned char),cudaMemcpyHostToDevice);

	//specify texture parameters
	struct cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = dev_texArray[i];

	//specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	//create texture object
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	cudaMemcpy(dev_texObjs+i,&texObj,sizeof(cudaTextureObject_t),cudaMemcpyHostToDevice);

	texObjs.push_back(texObj);
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

template <class T>
void mallocAndCopytoGPU(T*& d, std::vector<T>& h) {
	cudaMalloc(&d, h.size() * sizeof(T));
	cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	dev_final_paths = dev_paths;

	mallocAndCopytoGPU<Material>(dev_materials,scene->materials);
	mallocAndCopytoGPU<Geom>(dev_geoms, scene->geoms);
	mallocAndCopytoGPU<Mesh>(dev_meshes, scene->meshes);

	mallocAndCopytoGPU<Primitive>(dev_prim_data.primitives, scene->primitives);
	mallocAndCopytoGPU<uint16_t>(dev_prim_data.indices, scene->mesh_indices);
	mallocAndCopytoGPU<glm::vec3>(dev_prim_data.normals, scene->mesh_normals);
	mallocAndCopytoGPU<glm::vec2>(dev_prim_data.texCoords, scene->mesh_uvs);
	mallocAndCopytoGPU<glm::vec3>(dev_prim_data.vertices, scene->mesh_vertices);
	mallocAndCopytoGPU<glm::vec4>(dev_prim_data.tangents, scene->mesh_tangents);

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	cudaMalloc(&dev_denoised_img, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_img, 0, pixelcount * sizeof(glm::vec3));

	//Texture memory
	texObjs.clear();
	dev_texArray.clear();
	cudaMalloc(&dev_texObjs,scene->textures.size()*sizeof(cudaTextureObject_t));
	dev_texArray.resize(scene->textures.size());

	for (int i = 0; i < scene->textures.size(); i++)
	{
		textureInit(scene->textures[i], i);
	}


	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	thrust_dev_intersections = thrust::device_pointer_cast(dev_intersections);

	// TODO: initialize any extra device memeory you need

	thrust_dev_paths = thrust::device_pointer_cast(dev_paths);

	cudaMalloc(&dev_materialID, pixelcount * sizeof(int));
	cudaMemset(dev_materialID, 0, pixelcount * sizeof(int));
	thrust_dev_materialID = thrust::device_pointer_cast(dev_materialID);

	cudaMalloc(&dev_materialID_copy, pixelcount * sizeof(int));
	cudaMemset(dev_materialID_copy, 0, pixelcount * sizeof(int));
	thrust_dev_materialIDCpy = thrust::device_pointer_cast(dev_materialID_copy);


#if ENABLE_FIRST_INTERSECTION_CACHE
	cudaMalloc(&dev_first_intersect, pixelcount * sizeof(ShadeableIntersection));

	cudaMemset(dev_first_intersect,0,pixelcount*sizeof(ShadeableIntersection));
#endif

#if Timer
	cudaEventCreate(&startEvent);
	cudaEventCreate(&endEvent);
#endif

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
	cudaFree(dev_materialID);
	cudaFree(dev_materialID_copy);
	//cudaFree(dev_final_paths);
	cudaFree(dev_triangles);
	cudaFree(dev_denoised_img);
	dev_prim_data.free();


	for (int i = 0; i < texObjs.size(); i++) {
		cudaDestroyTextureObject(texObjs[i]);
		cudaFreeArray(dev_texArray[i]);
	}

#if ENABLE_FIRST_INTERSECTION_CACHE
	cudaFree(dev_first_intersect);
#endif

#if Timer
	if (startEvent != NULL)
		cudaEventDestroy(startEvent);
	if (endEvent != NULL)
		cudaEventDestroy(endEvent);
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

	bool enableDepthField = false;
	bool enableStochasticAA = false;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f);

		//SSAA anti-aliasing
		if (enableStochasticAA)
		{
			thrust::random::uniform_real_distribution<float> u0(-1.f, 1.f);
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x + u0(rng) - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y + u0(rng) - (float)cam.resolution.y * 0.5f));
		}
		else
	    {
		    segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
	    }
		//DepthField
		if (enableDepthField)
		{
			float lensRadius = 0.2f; 
			float focalDistance = 5.25f;
			thrust::normal_distribution<float> n01(0, 1);
			float theta = u01(rng) * TWO_PI;
			glm::vec3 circlePerturb = lensRadius * n01(rng) * (cos(theta) * cam.right + sin(theta) * cam.up);
			glm::vec3 originalDir = segment.ray.direction;
			float ft = focalDistance / glm::dot(originalDir, cam.view);
			segment.ray.origin = segment.ray.origin + circlePerturb;
			segment.ray.direction = glm::normalize(ft * originalDir - circlePerturb);
		}
		
		//segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		// TODO: implement antialiasing by jittering the ray
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer,Camera cam) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].t = shadeableIntersections[idx].t;
		gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
		glm::vec3 point = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
		gBuffer[idx].position = point;

		// get depth buffer value
		//point is in world space
		glm::vec4 worldPoint = glm::vec4(point, 1.f);

		glm::mat4 cameraTransform = cam.projMat * cam.viewMat;
		glm::vec4 pointEye = cameraTransform * worldPoint;


		pointEye /= pointEye.w; //In NDC

		//printf("PointEye NDC: %f , %f , %f , %f \n", pointEye.x, pointEye.y, pointEye.z, pointEye.w);

		float z = pointEye.z;
		z = z / 2 + 0.5f;
	
		gBuffer[idx].depth = z;
	}
}

__device__ glm::vec4 depthReconstructPos(float depth, glm::vec2 screenPos,const Camera cam)
{
	glm::vec4 ndc = glm::vec4(screenPos.x , screenPos.y , depth * 2 - 1, 1);
	glm::mat4 cameraMatrix = cam.projMat * cam.viewMat;
	// ndc -> world
	glm::vec4 worldPos = glm::inverse(cameraMatrix) * ndc;
	worldPos /= worldPos.w;
	return worldPos;
}

//Calculate the weight for gBuffer Data
__device__ float computeGBufferWeight(glm::vec3&p, glm::vec3& q,float phi)
{
	glm::vec3 disc_vec = p - q;
	float distance = glm::dot(disc_vec, disc_vec);

	float factor1 = exp((-distance) / phi+0.0001f);
	return min(factor1, 1.f);
}

__device__ float gaussianWeight(int x,int y,float s)
{
	float factor1 = 1.0f / (2 * PI * s * s);
	float factor2 = exp(-(x * x + y * y) / (2 * s * s));
}

__global__ void normalizeImage(int img_width,int img_height,glm::vec3* imageData,int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < img_width && y < img_height)
	{
		int index = x + (y * img_width);
		glm::vec3 pix = imageData[index];

		pix.x /= iter;
		pix.y /= iter;
		pix.z /= iter;

		imageData[index] = pix;
	}
}

__global__ void kernDenoise(int img_width,int img_height,glm::vec3* imageData,int filterSize,
	GBufferPixel* gBuffer,int stepWidth,float colorWeight,
	float normalWeight,float positionWeight,Camera cam)
{
	// 5x5 B3-spline fliter
	// Filter h is based on a B3 spline interpolation
	float kernel[5][5] = {
	  0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
	  0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
	  0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
	  0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
	  0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625 };

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int index = x + (y * img_width);

	if (index < img_width * img_height)
	{

		glm::vec3 c1 = glm::vec3(0.f);

		float k = 0.f;
		for (int i = -2; i <= 2; i++)
		{
			for (int j = -2; j <= 2; j++)
			{
				
				int x0 = x + i * stepWidth;
				int y0 = y + j * stepWidth;
				//check if x and y is within Bound
				if (x0 >= 0 && x0 < img_width && y0 >= 0 && y0 < img_height)
				{
					int idx = x0 + y0 * img_width;
					float weight = 1.f;

#if USE_GBUFFER_METHOD
					float c_w = computeGBufferWeight(imageData[index], imageData[idx], colorWeight);
					float n_w = computeGBufferWeight(gBuffer[index].normal, gBuffer[idx].normal, normalWeight);
#if USE_DEPTH_RECONSTRUCT
					glm::vec2 pixelCoord_0 = glm::vec2(x, y);
					glm::vec2 pixelCoord_1 = glm::vec2(x0, y0);

					float screenPos_0_x = (x / img_width) * 2 - 1;
					float screenPos_0_y = 1 - (y / img_height) * 2;

					float screenPos_1_x = (x0 / img_width) * 2 - 1;
					float screenPos_1_y = 1 - (y0 / img_height) * 2;

					glm::vec2 screenPos_0 = glm::vec2(screenPos_0_x, screenPos_0_y);
					glm::vec2 screenPos_1 = glm::vec2(screenPos_1_x, screenPos_1_y);

					glm::vec3 position_0 = glm::vec3(depthReconstructPos(gBuffer[index].depth, screenPos_0, cam));
					glm::vec3 position_1 = glm::vec3(depthReconstructPos(gBuffer[idx].depth, screenPos_1, cam));

					float p_w = computeGBufferWeight(position_0, position_1, positionWeight);

#endif
#if NOT_USE_DEPTH_RECONSTRUCT
					float p_w = computeGBufferWeight(gBuffer[index].position, gBuffer[idx].position, positionWeight);
#endif
					weight = c_w * n_w * p_w;
#endif
					
					float ker = kernel[i + 2][j + 2];
					c1 += weight * ker * imageData[idx];
					k += weight * ker;
				}
			}
		}
		imageData[index] = c1 / k;

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
	, Mesh* meshes
	, int geoms_size
	, PrimitiveData dev_prim_data
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

		glm::vec2 uv;
		glm::vec4 tangent;
		int materialID = -1;

		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
	

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
		glm::vec4 tmp_tangent;
		int temp_materialID = -1;

		// naive parse through global geoms
	//	printf("Mesh data debug index 0 1 2: %d %d %d \n", dev_prim_data.indices[0], dev_prim_data.indices[1], dev_prim_data.indices[2]);

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				temp_materialID = geom.materialid;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				temp_materialID = geom.materialid;
			}
			else if (geom.type == MESH)
			{

				t = meshIntersectionTest(geom, meshes[geom.mesh_id], dev_prim_data, pathSegment.ray,
					tmp_intersect, tmp_normal, tmp_uv, tmp_tangent,temp_materialID,pathSegment.color);
	
			//	std::cout << "Debug msg: mesh triggered" << std::endl;
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				materialID = temp_materialID;
				normal = tmp_normal;
				tangent = tmp_tangent;
				uv = tmp_uv;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}
		ShadeableIntersection& intersection = intersections[path_index];
		if (hit_geom_index == -1)
		{
			intersection.t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersection.t = t_min;
			intersection.materialId = materialID;
			intersection.surfaceNormal = normal;
			intersection.tangent = tangent;
			intersection.uv = uv;
			//Add here
			intersection.intersectionPoint = intersect_point;
			//printf("get intersections \n");
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



__global__ void BSDFShading(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, cudaTextureObject_t* textures
)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_paths)
	{

		ShadeableIntersection intersection = shadeableIntersections[index];
		if (intersection.t > 0.0f)
		{
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			//glm::vec3 materialColor = glm::vec3(material.pbrVal.baseColor);
			//Ray ends when ray hit the light
			if (material.emittance>0.0)
			{
				pathSegments[index].color *= (material.color * material.emittance);
				pathSegments[index].remainingBounces = 0;
			}
			else
			{
				//need intersection position
				glm::vec3 inter = getPointOnRay(pathSegments[index].ray, intersection.t);
				
				scatterRay(pathSegments[index],intersection, inter, intersection.surfaceNormal, material,textures,rng);
				//Debug
				pathSegments[index].remainingBounces -= 1;
			}
		}
		else
		{
			pathSegments[index].color = glm::vec3(0.f);
			pathSegments[index].remainingBounces = 0;
		}
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

void showGBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	GBufferMode mode = GBufferMode::Depth;

	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer,mode,cam);
}

void showImage(uchar4* pbo, int iter,bool denoise) 
{
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	if (!denoise)
	{
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	}
	else
	{
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, 1, dev_denoised_img);
	}
	
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

struct not_termintated
{
	__host__ __device__
		bool operator()(const PathSegment& p)
	{
		return p.remainingBounces > 0;
	}
};
void pathtrace(uchar4* pbo, int frame, int iter,bool sortMaterial,bool denoise,
	int filterSize,int filterPasses,float colorWeight,float normalWeight,float positionWeight) 
{

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
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
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
	// 
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	int depth = 0;

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
	cudaMemset(dev_gBuffer, 0, num_paths * sizeof(GBufferPixel));
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	ShadeableIntersection* intersections = NULL;

	while (!iterationComplete) 
	{
		    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if	ENABLE_FIRST_INTERSECTION_CACHE
			if (depth == 0 && iter != 1)
			{
				intersections = dev_first_intersect;
			}
#endif
				// tracing
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, dev_meshes
					, hst_scene->geoms.size()
					, dev_prim_data
					, dev_intersections
					);
				checkCUDAError("trace one bounce");

				cudaDeviceSynchronize();
				if (depth == 0)
				{
					generateGBuffer << <numblocksPathSegmentTracing,blockSize1d >> > (num_paths,dev_intersections,dev_paths,dev_gBuffer,cam);
				}

	
#if ENABLE_FIRST_INTERSECTION_CACHE
				if (depth == 0 && iter == 1)
				{
					cudaMemcpy(dev_first_intersect, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				}
#endif
				intersections = dev_intersections;
			depth++;

#if ENABLE_FIRST_INTERSECTION_CACHE
			if (depth == 0 && iter == 1)
				intersections = dev_first_intersect;
#endif

	// TODO:
	// --- Shading Stage ---
	 // Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
#if ENABLE_RAY_SORTING
		//sort the intersections
		thrust::sort_by_key(thrust::device,dev_intersections,dev_intersections+num_paths,dev_paths,compareIntersection());
#endif

#if Timer
		cudaEventRecord(startEvent);
#endif

	   BSDFShading << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			intersections,
			dev_paths,
			dev_materials,
		    dev_texObjs
			);
	   cudaDeviceSynchronize();

	   //String Compaction Here
	    dev_paths = thrust::stable_partition(thrust::device,dev_paths,dev_paths+num_paths,isPathCompleted());

		//num_paths was changed here
		num_paths = dev_path_end - dev_paths;
		iterationComplete = (num_paths == 0);
		intersections=NULL;


	
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}	
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//num_path already equal to zero
	//finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_final_paths);
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_final_paths);

	dev_paths = dev_final_paths;

	///////////////////////////////////////////////////////////////////////////

	// Render finished, now can do denoise
	if (denoise)
	{
		cudaMemcpy(dev_denoised_img,dev_image,pixelcount*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
		checkCUDAError("dev_denoise_img:");

		normalizeImage << <blocksPerGrid2d, blockSize2d >> > (cam.resolution.x,cam.resolution.y,dev_denoised_img,iter);

		for (int i = 0; i < filterPasses; i++)
		{
			int stepWidth = 1;
			while (4*stepWidth <= filterSize)
			{
				kernDenoise << <blocksPerGrid2d, blockSize2d >> > 
					(cam.resolution.x,
					 cam.resolution.y,
					 dev_denoised_img,filterSize,
					 dev_gBuffer,stepWidth,
					 colorWeight,normalWeight,positionWeight,cam);
				stepWidth <<= 1;
			}
		}
		cudaMemcpy(hst_scene->state.image.data(),dev_denoised_img,
			pixelcount*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}

#if Timer
	cudaEventRecord(endEvent);
	cudaEventSynchronize(endEvent);
	float ms;
	cudaEventElapsedTime(&ms, startEvent, endEvent);
	if (depth == 8) {
		std::cout << iter;
		std::cout << " " << depth;
		std::cout << " " << ms;
		std::cout << " " << num_paths << endl;
	}
#endif
	checkCUDAError("pathtrace");

	// Send results to OpenGL buffer for rendering
	//sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

}
