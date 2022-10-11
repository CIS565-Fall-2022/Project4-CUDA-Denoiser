#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

void InitDataContainer(GuiDataContainer* guiData);

void copyImageToPBO(uchar4* devPBO, glm::vec3* devImage, int width, int height, int toneMapping);

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(glm::vec3* devDirectIllum, glm::vec3* devIndirectIllum);