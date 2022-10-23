#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
//void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);

void applyDenoise(float c_phi, float n_phi, float p_phi, float filtersize, int iter);
void showDenoiseBuffer(uchar4* pbo);
void copyDevImage();
