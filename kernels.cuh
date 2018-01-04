#include <cuda_runtime.h>

struct edgeIndicatorSurface
{
    cudaSurfaceObject_t edgeSurface;
    cudaSurfaceObject_t edgeGradSurface;
};

__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom);
