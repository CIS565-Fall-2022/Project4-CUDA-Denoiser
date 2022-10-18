#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo, int bit, bool save);
void showImage(uchar4* pbo, bool denoise, int filterSize, float colorWeight, float posWeight, float normWeight, bool save);
