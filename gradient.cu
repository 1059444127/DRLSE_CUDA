#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <gradient.cuh>
#include <common.cuh>


__global__ void sobelKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
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
            surf2Dread(&sample, input, (x+i)*sizeof(sample), y+j, cudaBoundaryModeClamp);

            index = 5*(i+2) + (j+2);
            sumX += sample * d_sobelX[index];
            sumY += sample * d_sobelY[index];
        }
    }

    surf2Dwrite<float2>(make_float2(sumX, sumY),
                        output, x * sizeof(float2),
                        y,
                        cudaBoundaryModeClamp);
}

__host__ float* applySobelFilter(int imageWidth, int imageHeight, float* h_dicomData)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(float);

    // Create a Surface with our image data and copy that data to the device
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelFormatGrad = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    auto dicomSurf = CUDASurface(h_dicomData, imageWidth, imageHeight, channelFormatDicom);
    auto gradSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatGrad);

    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    sobelKernel<<<grid, block>>>(dicomSurf.surface, gradSurf.surface);

    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Copy results to host
    float* h_gradData = (float*)malloc(sizeDicom * 2);
    eee(cudaMemcpyFromArray(h_gradData, gradSurf.arr, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    return h_gradData;
}
