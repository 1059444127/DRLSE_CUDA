#include <cuda_runtime.h>

__global__ void gaussianKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__host__ float* applyGaussianFilter(int imageWidth, int imageHeight, float* h_dicomData);
