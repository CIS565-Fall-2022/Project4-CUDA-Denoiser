#pragma once

#include <vector>
#include "scene.h"
#include "main.h"

void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void denoise();
