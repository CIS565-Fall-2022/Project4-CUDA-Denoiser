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

    glm::vec3* devAlbedo = nullptr;
    glm::vec3* devNormal = nullptr;
    float* devDepth = nullptr;
    int* devPrimId = nullptr;
};

void denoiserInit(int width, int height);
void denoiserFree();

void renderGBuffer(DevScene* scene, Camera cam, GBuffer gBuffer);

void accumlateVariance();

void waveletFilter(GBuffer gBuffer);

void modulateAlbedo(glm::vec3* devImage, GBuffer gBuffer, int width, int height);