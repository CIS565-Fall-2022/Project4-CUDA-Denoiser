#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, int type);
void showImage(uchar4 *pbo, int iter);
void showImageDenoise(uchar4* pbo, int iter);
