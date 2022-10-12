#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common.h"
#include "scene.h"
#include "pathtrace.h"

struct GBuffer {
    GBuffer() = default;

    void create(int width, int height);
    void destroy();
    void render(DevScene* scene, const Camera& cam);
    void update(const Camera& cam);

    __host__ __device__ glm::vec3* normal() { return devNormal[frame]; }
    __host__ __device__ int* primId() { return devPrimId[frame]; }
    __host__ __device__ float* depth() { return devDepth[frame]; }

    glm::vec3* devAlbedo = nullptr;
    glm::vec2* devMotion = nullptr;
    glm::vec3* devNormal[2] = { nullptr };
    float* devDepth[2] = { nullptr };
    int* devPrimId[2] = { nullptr };
    int frame = 0;

    Camera lastCamera;
};

struct EAWaveletFilter {
    EAWaveletFilter() = default;

    EAWaveletFilter(int width, int height) :
        width(width), height(height) {}

    void filter(glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level);
    void filter(glm::vec4* devColorVarOut, glm::vec4* devColorVarIn, const GBuffer& gBuffer, const Camera& cam, int level);

    float sigLumin = 64.f;
    float sigNormal = .2f;
    float sigDepth = 1.f;

    int width;
    int height;
};

struct LeveledEAWFilter {
    LeveledEAWFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void filter(glm::vec3*& devColorIn, const GBuffer& gBuffer, const Camera& cam);

    EAWaveletFilter waveletFilter;
    int level = 0;
    glm::vec3* devTempImg = nullptr;
};

struct SVGFFilter {
    EAWaveletFilter waveletFilter;
    int level = 0;
    glm::vec4* devTempColorVar = nullptr;
};

void denoiserInit(int width, int height);
void denoiserFree();

void modulateAlbedo(glm::vec3* devImage, GBuffer gBuffer, int width, int height);
void addImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height);