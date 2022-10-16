#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration,bool sortMaterial);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);