#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <gaussian.cuh>
#include <common.cuh>


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

__host__ float* applyGaussianFilter(int imageWidth, int imageHeight, float* h_dicomData)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(float);

    // Create a Surface with our image data and copy that data to the device
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    auto dicomSurf = CUDASurface(h_dicomData, imageWidth, imageHeight, channelFormatDicom);
    auto gaussSurf = CUDASurface(nullptr, imageWidth, imageHeight, channelFormatDicom);

    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    gaussianKernel<<<grid, block>>>(dicomSurf.surface, gaussSurf.surface);

    // The synchronize call will force the host to wait for the kernel to finish. If we don't
    // do this, we might get errors on future checks, but that indicate errors in the kernel, which
    // can be confusing
    eee(cudaPeekAtLastError());
    eee(cudaDeviceSynchronize());

    // Copy results to host
    float* h_gaussData = (float*)malloc(sizeDicom);
    eee(cudaMemcpyFromArray(h_gaussData, gaussSurf.arr, 0, 0, sizeDicom, cudaMemcpyDeviceToHost));

    return h_gaussData;
}
