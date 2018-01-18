#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <memory>

#include <kernels.cuh>
#include <gaussian.cuh>
#include <gradient.cuh>
#include <common.cuh>

using namespace std;

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

//Forward declarations
__host__ void edgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom);


__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(float);

    // Run the kernel, generating edge indicator and its gradient as surfaces
    shared_ptr<CUDASurface> edge, edgeGrad;
    edgeIndicator(imageWidth, imageHeight, h_dataDicom, edge.get(), edgeGrad.get());

    // Copy results to host memory
    float* h_output = (float*)malloc(sizeDicom);
    eee(cudaMemcpyFromArray(h_output, edge->arr, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    // Cleanup
    eee(cudaDeviceReset());

    return h_output;
}

__host__ void edgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom)
{
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatGauss = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatEdge = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatEdgeGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

    auto dicomSurf = CUDASurface(h_dataDicom, imageWidth, imageHeight, channelFormatDicom);
    auto gaussSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatGauss);
    out_edgeSurf = new CUDASurface(nullptr, imageWidth, imageHeight, channelFormatEdge);
    out_edgeGradSurf = new CUDASurface(nullptr, imageWidth, imageHeight, channelFormatEdgeGrad);

    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);

    // Run gaussian kernel
    gaussianKernel<<<grid, block>>>(dicomSurf.surface, gaussSurf.surface);

    // Run edge indicator kernel
    edgeIndicatorKernel<<<grid, block>>>(gaussSurf.surface, out_edgeSurf->surface);

    // Also get the gradient of the edge indicator result
    sobelKernel<<<grid, block>>>(out_edgeSurf->surface, out_edgeGradSurf->surface);


    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Cleanup    
    eee(cudaDeviceReset());
}
