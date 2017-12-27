#include <cuda_runtime.h>

//This file gets included by the cpp translation unit as well as the cuda compiler
//these definitions allow us to use these templates anywhere. They will need to be
//explicitly instantiated at the bottom of kernels.cu however
template<typename T, cudaChannelFormatKind FK>
__host__ T* modifyTexture(int imageWidth, int imageHeight, T* textureData);

template<typename T, cudaChannelFormatKind FK>
__host__ T* modifyTextureRasterized(int imageWidth, int imageHeight, T* dicomData, unsigned char* polylineData);

template<typename T, cudaChannelFormatKind FK>
__host__ T* modifySurfaceRasterized(int imageWidth, int imageHeight, T* dicomData, unsigned char* polylineData);

template<typename T, cudaChannelFormatKind FK>
__host__ T* convolveImage(int imageWidth, int imageHeight, T* textureData);
