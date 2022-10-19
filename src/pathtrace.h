#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void denoise(float c_phi, float n_phi, float p_phi, float filterSize);

void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoised(uchar4* pbo, int iter);