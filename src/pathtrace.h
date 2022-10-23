#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void denoise(int iter, int filterSize = 0, float colorWeight = 0.f, float norWeight = 0.f, float posWeight = 0.f);
void showGBuffer(uchar4* pbo, const string displayedData);
void showImage(uchar4* pbo, int iter, bool isDenoise);
void printDenoiseTime();
