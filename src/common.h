#pragma once
#include <iostream>

#define SAMPLER_USE_SOBOL true

#define SCENE_LIGHT_SINGLE_SIDED true

#define DENOISER_DEMODULATE true
#define DENOISER_ENCODE_NORMAL true
#define DENOISER_ENCODE_POSITION false

#define DEMODULATE_EPS 1e-3f

#define DenoiseCompress 1.f
#define DenoiseLightId -2

struct ToneMapping {
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
};

struct Tracer {
    enum {
        Streamed = 0, SingleKernel = 1, BVHVisualize = 2, GBufferPreview = 3
    };
};

struct Denoiser {
    enum {
        None, Gaussian, EAWavelet, SVGF
    };
};

struct Settings {
    static int traceDepth;
    static int toneMapping;
    static int tracer;
    static int ImagePreviewOpt;
    static int denoiser;
    static bool modulate;
};

struct State {
    static bool camChanged;
};
