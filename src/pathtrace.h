#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, bool useZforPos);
void pathtraceFree(bool useZforPos);
void pathtrace(int frame, int iteration, int filterSeize, float colorWeight, float normalWeight, float positionWeight, bool denoiser, bool useZforPos);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void setKernelOffset(float* dev_kernel, glm::ivec2* dev_offset);
