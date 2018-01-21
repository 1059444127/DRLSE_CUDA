#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <string>

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

class CUDASurface
{
public:
    std::string name;

    cudaSurfaceObject_t surface;
    cudaResourceDesc surfDesc;
    cudaArray* arr;
    cudaChannelFormatDesc arrDesc;    

    CUDASurface(const void* h_src, unsigned int width, unsigned int height, const cudaChannelFormatDesc& desc)
    {
        //Array description and memcpy
        arrDesc = desc;
        eee(cudaMallocArray(&arr, &arrDesc, width, height));
        if(h_src != NULL)
            eee(cudaMemcpyToArray(arr, 0, 0, h_src, width * height * ((desc.x / 8) + (desc.y / 8) + (desc.z / 8) + (desc.w / 8)), cudaMemcpyHostToDevice));

        //Surface description
        memset(&surfDesc, 0, sizeof(surfDesc));
        surfDesc.resType = cudaResourceTypeArray;
        surfDesc.res.array.array = arr;

        //Surface object
        memset(&surface, 0, sizeof(surface));
        eee(cudaCreateSurfaceObject(&surface, &surfDesc));
    }

    ~CUDASurface()
    {
        eee(cudaDestroySurfaceObject(surface));
        eee(cudaFreeArray(arr));
    }
};
