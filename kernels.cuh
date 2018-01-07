#include <cuda_runtime.h>

struct edgeIndicatorSurface
{
    cudaSurfaceObject_t edge;
    cudaSurfaceObject_t edgeGrad;
};

__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom);
