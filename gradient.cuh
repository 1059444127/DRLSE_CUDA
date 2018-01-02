#include <cuda_runtime.h>

//This file gets included by the cpp translation unit as well as the cuda compiler
//these definitions allow us to use these templates anywhere. They will need to be
//explicitly instantiated at the bottom of gradient.cu however

template<typename T, cudaChannelFormatKind FK>
__host__ float* applySobelFilter(int imageWidth, int imageHeight, T* h_dataDicom);
