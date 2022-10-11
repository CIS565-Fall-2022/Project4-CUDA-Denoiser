#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::ACES;
int Settings::tracer = Tracer::Streamed;
int Settings::ImagePreviewOpt = 3;

bool State::camChanged = true;