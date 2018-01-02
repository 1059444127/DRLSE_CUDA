#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <gaussian.cuh>
#include <common.cuh>

__constant__ float d_identity[5*5] =   {0,   0,   0,   0,  0,
                                        0,   0,   0,   0,  0,
                                        0,   0,   1,   0,  0,
                                        0,   0,   0,   0,  0,
                                        0,   0,   0,   0,  0};

__constant__ float d_gaussKernel3[3*3];
__constant__ float d_gaussKernel5[5*5] = {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
                                          0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
                                          0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
                                          0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
                                          0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f};

__global__ void gaussianKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    float sample;

    for(int i = -2; i <= 2; i++)
    {
        for(int j = -2; j <= 2; j++)
        {
            surf2Dread(&sample, input, (x+i)*sizeof(sample), y+j, cudaBoundaryModeClamp);
            sum += sample * d_gaussKernel5[5*(i+2) + (j+2)];
        }
    }

    surf2Dwrite(sum,
                output, x * sizeof(float),
                y,
                cudaBoundaryModeClamp);
}

__host__ float* applyGaussianFilter(int imageWidth, int imageHeight, float* h_dataDicom)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(float);

    // Create a Surface with our image data and copy that data to the device
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayDicom;
    eee(cudaMallocArray(&d_arrayDicom, &channelFormatDicom, imageWidth, imageHeight));
    eee(cudaMemcpyToArray(d_arrayDicom, 0, 0, h_dataDicom, sizeDicom, cudaMemcpyHostToDevice));

    cudaResourceDesc resDescDicom;
    memset(&resDescDicom, 0, sizeof(resDescDicom));
    resDescDicom.resType = cudaResourceTypeArray;
    resDescDicom.res.array.array = d_arrayDicom;

    cudaSurfaceObject_t d_surfDicom = 0;
    eee(cudaCreateSurfaceObject(&d_surfDicom, &resDescDicom));


    // Create an output surface
    cudaArray* d_arrayResult;
    eee(cudaMallocArray(&d_arrayResult, &channelFormatDicom, imageWidth, imageHeight));

    cudaResourceDesc resDescResult;
    memset(&resDescResult, 0, sizeof(resDescResult));
    resDescResult.resType = cudaResourceTypeArray;
    resDescResult.res.array.array = d_arrayResult;

    cudaSurfaceObject_t d_surfResult = 0;
    eee(cudaCreateSurfaceObject(&d_surfResult, &resDescResult));


    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    gaussianKernel<<<grid, block>>>(d_surfDicom, d_surfResult);

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
