#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void runDenoiser(int filterSize, float colorWeight, float normalWeight, float positionWeight);
void showDenoisedImage(uchar4* pbo, int iter);
