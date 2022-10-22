#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void denoise(float cw, float nw, float pw, int filterSize, int iter, int ui_iterations);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter, bool denoise);