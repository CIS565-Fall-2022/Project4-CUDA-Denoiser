#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void denoise(uchar4* pbo, int iter, float sigma_col, float sigma_norm, float sigma_pos);
void showImage(uchar4 *pbo, int iter);
