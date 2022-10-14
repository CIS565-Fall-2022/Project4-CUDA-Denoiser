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
    int* devMotion = nullptr;
    glm::vec3* devNormal[2] = { nullptr };
    float* devDepth[2] = { nullptr };
    int* devPrimId[2] = { nullptr };
    int frame = 0;

    Camera lastCamera;
    int width;
    int height;
};

struct EAWaveletFilter {
    EAWaveletFilter() = default;

    EAWaveletFilter(int width, int height, float sigLumin, float sigNormal, float sigDepth) :
        width(width), height(height), sigLumin(sigLumin), sigNormal(sigNormal), sigDepth(sigDepth) {}

    void filter(glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level);
    void filter(glm::vec3* devColorOut, glm::vec3* devColorIn, float* devVarianceOut, float* devVarianceIn,
        float* devFilteredVar, const GBuffer& gBuffer, const Camera& cam, int level);

    float sigLumin;
    float sigNormal;
    float sigDepth;

    int width = 0;
    int height = 0;
};

struct LeveledEAWFilter {
    LeveledEAWFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void filter(glm::vec3*& devColor, const GBuffer& gBuffer, const Camera& cam);

    EAWaveletFilter waveletFilter;
    int level = 0;
    glm::vec3* devTempImg = nullptr;
};

struct SpatioTemporalFilter {
    SpatioTemporalFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void temporalAccumulate(glm::vec3* devColorIn, const GBuffer& gBuffer);
    void estimateVariance();
    void filterVariance();

    void filter(glm::vec3*& devColor, const GBuffer& gBuffer, const Camera& cam);

    EAWaveletFilter waveletFilter;
    int level = 0;

    glm::vec3* devAccumColor = nullptr;
    glm::vec3* devAccumMoment = nullptr;
    float* devVariance = nullptr;
    bool firstTime = true;

    glm::vec3* devTempColor = nullptr;
    float* devTempVariance = nullptr;
    float* devFilteredVariance = nullptr;
};

void denoiserInit(int width, int height);
void denoiserFree();

void modulateAlbedo(glm::vec3* devImage, const GBuffer& gBuffer);
void addImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height);
void addImage(glm::vec3* devOut, glm::vec3* devIn1, glm::vec3* devIn2, int width, int height);