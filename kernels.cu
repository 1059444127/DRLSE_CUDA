#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <kernels.cuh>
#include <gaussian.cuh>
#include <common.cuh>

//====================================================================================
//KERNELS
//====================================================================================

__global__ void edgeIndicatorKernel(cudaSurfaceObject_t gaussInput, cudaSurfaceObject_t output)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sumX = 0;
    float sumY = 0;
    int index = 0;
    float sample;

    for(int i = -2; i <= 2; i++)
    {
        for(int j = -2; j <= 2; j++)
        {
            surf2Dread(&sample, gaussInput, (x+i)*sizeof(sample), y+j, cudaBoundaryModeClamp);

            index = 5*(i+2) + (j+2);
            sumX += sample * d_sobelX[index];
            sumY += sample * d_sobelY[index];
        }
    }

    sumX = sumX / 1000.0f;
    sumY = sumY / 1000.0f;

    surf2Dwrite(1.0f / (1.0f + sumX * sumX + sumY * sumY),
                output, x * sizeof(float),
                y,
                cudaBoundaryModeClamp);
}

//====================================================================================
//HOST CUDA FUNCTIONS
//====================================================================================
__host__ float* testEdgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom)
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


    // Create a temp surface for the gaussian filtered input image
    cudaChannelFormatDesc channelFormatGauss = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayGauss;
    eee(cudaMallocArray(&d_arrayGauss, &channelFormatGauss, imageWidth, imageHeight));

    cudaResourceDesc resDescGauss;
    memset(&resDescGauss, 0, sizeof(resDescGauss));
    resDescGauss.resType = cudaResourceTypeArray;
    resDescGauss.res.array.array = d_arrayGauss;

    cudaSurfaceObject_t d_surfGaussian = 0;
    eee(cudaCreateSurfaceObject(&d_surfGaussian, &resDescGauss));


    // Create an output surface for the edge indicator image
    cudaChannelFormatDesc channelFormatRes = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayResult;
    eee(cudaMallocArray(&d_arrayResult, &channelFormatRes, imageWidth, imageHeight));

    cudaResourceDesc resDescResult;
    memset(&resDescResult, 0, sizeof(resDescResult));
    resDescResult.resType = cudaResourceTypeArray;
    resDescResult.res.array.array = d_arrayResult;

    cudaSurfaceObject_t d_surfResult = 0;
    eee(cudaCreateSurfaceObject(&d_surfResult, &resDescResult));


    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);

    // Run gaussian kernel
    gaussianKernel<<<grid, block>>>(d_surfDicom, d_surfGaussian);

    // Run kernel
    edgeIndicatorKernel<<<grid, block>>>(d_surfGaussian, d_surfResult);


    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Copy results to host
    float* outputHost = (float*)malloc(sizeDicom); //return 2 channels
    eee(cudaMemcpyFromArray(outputHost, d_arrayResult, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    // Cleanup
    eee(cudaDestroySurfaceObject(d_surfDicom));
    eee(cudaDestroySurfaceObject(d_surfGaussian));
    eee(cudaDestroySurfaceObject(d_surfResult));
    eee(cudaFreeArray(d_arrayDicom));
    eee(cudaFreeArray(d_arrayGauss));
    eee(cudaFreeArray(d_arrayResult));
    eee(cudaDeviceReset());

    return outputHost;
}
