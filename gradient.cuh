#include <cuda_runtime.h>

__global__ void sobelKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__host__ float* applySobelFilter(int imageWidth, int imageHeight, float* h_dicomData);
