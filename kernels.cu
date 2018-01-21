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
__host__ void edgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom, CUDASurface* out_edgeSurf, CUDASurface* out_edgeGradSurf);


__host__ float* applyEdgeIndicator(int imageWidth, int imageHeight, float* h_dicomData)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(float);

    cudaChannelFormatDesc channelFormatEdge = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatEdgeGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    auto edgeSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatEdge);
    auto edgeGradSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatEdgeGrad);
    edgeSurf.name = "edgeSurf";
    edgeGradSurf.name = "edgeGradSurf";

    edgeIndicator(imageWidth, imageHeight, h_dicomData, &edgeSurf, &edgeGradSurf);

    // Copy results to host memory
    float* h_output = (float*)malloc(sizeDicom);
    eee(cudaMemcpyFromArray(h_output, edgeSurf.arr, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    return h_output;
}

__host__ void edgeIndicator(int imageWidth, int imageHeight, float* h_dataDicom, CUDASurface *out_edgeSurf, CUDASurface *out_edgeGradSurf)
{
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatGauss = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    auto dicomSurf = CUDASurface(h_dataDicom, imageWidth, imageHeight, channelFormatDicom);
    auto gaussSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatGauss);
    dicomSurf.name = "dicomSurf";
    gaussSurf.name = "gaussSurf";

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
}

__host__ void initLevelSetData(int imageWidth, int imageHeight, float* h_dicomData, float* h_polylineData, LevelSetData* out_levelSetData)
{
    cudaChannelFormatDesc channelFormatPhi = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    out_levelSetData->phi = std::make_unique<CUDASurface>(h_polylineData, imageWidth, imageHeight, channelFormatPhi);
    out_levelSetData->phi->name = "phiSurf";

    cudaChannelFormatDesc channelFormatEdge = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    out_levelSetData->edge = std::make_unique<CUDASurface>(nullptr, imageWidth, imageHeight, channelFormatEdge);
    out_levelSetData->edge->name = "edgeSurf";

    cudaChannelFormatDesc channelFormatEdgeGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    out_levelSetData->edgeGrad = std::make_unique<CUDASurface>(nullptr, imageWidth, imageHeight, channelFormatEdgeGrad);
    out_levelSetData->edgeGrad->name = "edgeGradSurf";

    edgeIndicator(imageWidth, imageHeight, h_dicomData, out_levelSetData->edge.get(), out_levelSetData->edgeGrad.get());

    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());
}
