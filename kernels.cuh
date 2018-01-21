#include <cuda_runtime.h>

#include "common.cuh"

__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dicomData);
__host__ void initLevelSetData(int imageWidth, int imageHeight,
                               float* h_dicomData, float* h_polylineData,
                               LevelSetData* out_levelSetData);
