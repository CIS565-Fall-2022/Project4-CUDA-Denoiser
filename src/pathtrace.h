#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void denoise(float sigma_c, float sigma_n, float sigma_x, int filterSize);
void showDenoisedImage(uchar4* pbo, int iter);
void update();
