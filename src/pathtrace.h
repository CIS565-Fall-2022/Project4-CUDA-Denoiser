#pragma once

#include <vector>
#include "scene.h"

struct DenoiseParams {
	bool denoise;
	float sigma_p;
	float sigma_n;
	float sigma_rt;
};

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void denoiseAndShowImage(uchar4* pbo, int iter, DenoiseParams denoise_params);
