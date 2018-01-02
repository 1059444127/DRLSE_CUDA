#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <gradient.cuh>
#include <common.cuh>

__constant__ float d_sobelX[5*5] = {1,   2,   0,  -2,  -1,
                                    4,   8,   0,  -8,  -4,
                                    6,  12,   0, -12,  -6,
                                    4,   8,   0,  -8,  -4,
                                    1,   2,   0,  -2,  -1};

__constant__ float d_sobelY[5*5] = {1,   4,   6,   4,   1,
                                    2,   8,  12,   8,   2,
                                    0,   0,   0,   0,   0,
                                   -2,  -8, -12,  -8,  -2,
                                   -1,  -4,  -6,  -4,  -1};

template<typename T>
__global__ void sobelKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sumX = 0;
    float sumY = 0;
    int index = 0;
    T sample;

    for(int i = -2; i <= 2; i++)
    {
        for(int j = -2; j <= 2; j++)
        {
            surf2Dread(&sample, input, (x+i)*sizeof(sample), y+j, cudaBoundaryModeClamp);

            index = 5*(i+2) + (j+2);
            sumX += sample * d_sobelX[index];
            sumY += sample * d_sobelY[index];
        }
    }

    surf2Dwrite(make_float2(sumX, sumY),
                output, x * sizeof(float2),
                y,
                cudaBoundaryModeClamp);
}

template<typename T, cudaChannelFormatKind FK>
__host__ float* applySobelFilter(int imageWidth, int imageHeight, T* h_dataDicom)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(T);

    // Create a Surface with our image data and copy that data to the device
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(8 * sizeof(T), 0, 0, 0, FK);
    cudaArray* d_arrayDicom;
    eee(cudaMallocArray(&d_arrayDicom, &channelFormatDicom, imageWidth, imageHeight));
    eee(cudaMemcpyToArray(d_arrayDicom, 0, 0, h_dataDicom, sizeDicom, cudaMemcpyHostToDevice));

    cudaResourceDesc resDescDicom;
    memset(&resDescDicom, 0, sizeof(resDescDicom));
    resDescDicom.resType = cudaResourceTypeArray;
    resDescDicom.res.array.array = d_arrayDicom;

    cudaSurfaceObject_t d_surfDicom = 0;
    eee(cudaCreateSurfaceObject(&d_surfDicom, &resDescDicom));


    // Create an output surface, 32-bit float for x, y and magnitude
    cudaChannelFormatDesc channelFormatGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayResult;
    eee(cudaMallocArray(&d_arrayResult, &channelFormatGrad, imageWidth, imageHeight));

    cudaResourceDesc resDescResult;
    memset(&resDescResult, 0, sizeof(resDescResult));
    resDescResult.resType = cudaResourceTypeArray;
    resDescResult.res.array.array = d_arrayResult;

    cudaSurfaceObject_t d_surfResult = 0;
    eee(cudaCreateSurfaceObject(&d_surfResult, &resDescResult));


    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    sobelKernel<T> <<<grid, block>>>(d_surfDicom, d_surfResult);

    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Copy results to host
    float* outputHost = (float*)malloc(sizeDicom);
    eee(cudaMemcpyFromArray(outputHost, d_arrayResult, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    // Cleanup
    eee(cudaDestroySurfaceObject(d_surfDicom));
    eee(cudaDestroySurfaceObject(d_surfResult));
    eee(cudaFreeArray(d_arrayDicom));
    eee(cudaFreeArray(d_arrayResult));
    eee(cudaDeviceReset());

    return outputHost;
}

//Explicit instantiation since the compiler has no idea these will be needed in other compilation units
template __host__ float* applySobelFilter<short, cudaChannelFormatKindSigned>(int imageWidth, int imageHeight, short* textureData);
template __host__ float* applySobelFilter<unsigned short, cudaChannelFormatKindUnsigned>(int imageWidth, int imageHeight, unsigned short* textureData);
