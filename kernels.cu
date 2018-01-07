#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <kernels.cuh>
#include <gaussian.cuh>
#include <gradient.cuh>
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

    surf2Dwrite(1.0f / (1.0f + sumX * sumX + sumY * sumY),
                output, x * sizeof(float),
                y,
                cudaBoundaryModeClamp);
}

//====================================================================================
//HOST CUDA FUNCTIONS
//====================================================================================
__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom)
{
    // Run the kernel, generating edge indicator and its gradient as surfaces
    auto edgeSurf = edgeIndicator(imageWidth, imageHeight, dataDicom);

    // Get access to edge indicator's array
    cudaResourceDesc resDescEdge;
    eee(cudaGetSurfaceObjectResourceDesc(&resDescEdge, edgeSurf.edge));
    auto edgeCudaArray = resDescEdge.res.array.array;

    // Get access to edge indicator gradient's array
    cudaResourceDesc resDescEdgeGrad;
    eee(cudaGetSurfaceObjectResourceDesc(&resDescEdgeGrad, edgeSurf.edgeGrad));
    auto edgeGradCudaArray = resDescEdgeGrad.res.array.array;

    // Copy results to host
    float* outputHost = (float*)malloc(sizeDicom);
    eee(cudaMemcpyFromArray(outputHost, edgeCudaArray, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    // Cleanup
    eee(cudaDestroySurfaceObject(edgeSurf.edge));
    eee(cudaDestroySurfaceObject(edgeSurf.edgeGrad));
    eee(cudaFreeArray(edgeCudaArray));
    eee(cudaFreeArray(edgeGradCudaArray));
    eee(cudaDeviceReset());

    return outputHost;
}

__host__ edgeIndicatorSurface edgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom)
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
    cudaChannelFormatDesc channelFormatEdge = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayEdge;
    eee(cudaMallocArray(&d_arrayEdge, &channelFormatEdge, imageWidth, imageHeight));

    cudaResourceDesc resDescEdge;
    memset(&resDescEdge, 0, sizeof(resDescEdge));
    resDescEdge.resType = cudaResourceTypeArray;
    resDescEdge.res.array.array = d_arrayEdge;

    cudaSurfaceObject_t d_surfEdge = 0;
    eee(cudaCreateSurfaceObject(&d_surfEdge, &resDescEdge));


    // Create an output surface for the gradient of the edge indicator image
    cudaChannelFormatDesc channelFormatEdgeGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* d_arrayEdgeGrad;
    eee(cudaMallocArray(&d_arrayEdgeGrad, &channelFormatEdgeGrad, imageWidth, imageHeight));

    cudaResourceDesc resDescEdgeGrad;
    memset(&resDescEdgeGrad, 0, sizeof(resDescEdgeGrad));
    resDescEdgeGrad.resType = cudaResourceTypeArray;
    resDescEdgeGrad.res.array.array = d_arrayEdgeGrad;

    cudaSurfaceObject_t d_surfEdgeGrad = 0;
    eee(cudaCreateSurfaceObject(&d_surfEdgeGrad, &resDescEdgeGrad));


    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);

    // Run gaussian kernel
    gaussianKernel<<<grid, block>>>(d_surfDicom, d_surfGaussian);

    // Run edge indicator kernel
    edgeIndicatorKernel<<<grid, block>>>(d_surfGaussian, d_surfEdge);

    // Also get the gradient of the edge indicator result
    sobelKernel<<<grid, block>>>(d_surfEdge, d_surfEdgeGrad);


    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Cleanup
    eee(cudaDestroySurfaceObject(d_surfDicom));
    eee(cudaDestroySurfaceObject(d_surfGaussian));
    eee(cudaFreeArray(d_arrayDicom));
    eee(cudaFreeArray(d_arrayGauss));
    eee(cudaDeviceReset());

    edgeIndicatorSurface result;
    result.edge = d_surfEdge;
    result.edgeGrad = d_surfEdgeGrad;
    return result;
}
