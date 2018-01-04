#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

//Convolution kernels defined in common.cu
extern __constant__ float d_sobelX[5*5];
extern __constant__ float d_sobelY[5*5];
extern __constant__ float d_identity[5*5];
extern __constant__ float d_gaussKernel3[3*3];
extern __constant__ float d_gaussKernel5[5*5];
extern __constant__ float d_laplace3[3*3];
extern __constant__ float d_laplace5[5*5];

#define eee(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
