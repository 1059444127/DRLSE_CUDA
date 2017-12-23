#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void diagKernel(short* output, cudaTextureObject_t input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    output[y * width + x] = (short)(0.5 * tex2D<short>(input, u, v));
}

__global__ void diagKernel(unsigned short* output, cudaTextureObject_t input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    output[y * width + x] = (short)(0.5 * tex2D<unsigned short>(input, u, v));
}

__host__ short* modifyTexture(int imageWidth, int imageHeight, short* textureData)
{
    cudaError_t e;
    size_t size = imageWidth * imageHeight * sizeof(short);

    // Allocate a cudaArray in device memory and copy our texture data there
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArray, 0, 0, textureData, size, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    // Create result array in the device
    short* outputDev;
    cudaMalloc(&outputDev, size);

    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    diagKernel<<<grid, block>>>(outputDev, tex, imageWidth, imageHeight);

    // Create result array in the host
    short* outputHost = (short*)malloc(size);

    // Copy results to host
    cudaMemcpy(outputHost, outputDev, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cuArray);
    cudaFree(outputDev);

    return outputHost;
}

__host__ unsigned short* modifyTexture(int imageWidth, int imageHeight, unsigned short* textureData)
{
    cudaError_t e;
    size_t size = imageWidth * imageHeight * sizeof(unsigned short);

    // Allocate a cudaArray in device memory and copy our texture data there
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArray, 0, 0, textureData, size, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    // Create result array in the device
    unsigned short* outputDev;
    cudaMalloc(&outputDev, size);

    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    diagKernel<<<grid, block>>>(outputDev, tex, imageWidth, imageHeight);

    // Create result array in the host
    unsigned short* outputHost = (unsigned short*)malloc(size);

    // Copy results to host
    cudaMemcpy(outputHost, outputDev, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cuArray);
    cudaFree(outputDev);

    return outputHost;
}
