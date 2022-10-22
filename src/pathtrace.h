#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, float&timer);

void showGBuffer(uchar4* pbo, int mode);
void showImage(uchar4* pbo, int iter);

// Textures load and free
void LoadTexturesToDevice(Scene* scene);
void FreeTextures();

// BVH load and free
void LoadBVHToDevice(Scene* scene);
void FreeBVH();

// Skybox
void LoadSkyboxTextureToDevice(Scene* scene);
void FreeSkyboxTexure();

// Denoiser
void denoise(Scene* scene, float c_phi, float n_phi, float p_phi, float filterSize, float& timer);
void showDenoiseImage(uchar4* pbo, int iter);

void gaussianBlur(float filterSize);