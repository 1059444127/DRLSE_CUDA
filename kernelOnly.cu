#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void simpleKernel(short* output, cudaTextureObject_t input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    output[y * width + x] = (short)(0.5 * tex2D<short>(input, u, v));
}

__global__ void simpleKernel(unsigned short* output, cudaTextureObject_t input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    output[y * width + x] = (short)(0.5 * tex2D<unsigned short>(input, u, v));
}

__global__ void rasterizerTest(short* output, cudaTextureObject_t dicomTex, cudaTextureObject_t polylineTex, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    unsigned int sum = 0;

    //Set sum to 1 if we're next to a painted square on polylineTex
    short dicomSample = tex2D<short>(dicomTex, u, v);
    for(int i = -1; i < 2; i++)
    {
        for(int j = -1; j < 2; j++)
        {
            sum += tex2D<unsigned char>(polylineTex, (x + i) / (float)width, (y + j) / (float)height);
        }
    }
    sum = min(sum, 1);

    output[y * width + x] = dicomSample * (1 - sum);
}

__global__ void rasterizerTest(unsigned short* output, cudaTextureObject_t dicomTex, cudaTextureObject_t polylineTex, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    unsigned int sum = 0;

    //Set sum to 1 if we're next to a painted square on polylineTex
    unsigned short dicomSample = tex2D<unsigned short>(dicomTex, u, v);
    for(int i = -1; i < 2; i++)
    {
        for(int j = -1; j < 2; j++)
        {
            sum += tex2D<unsigned char>(polylineTex, (x + i) / (float)width, (y + j) / (float)height);
        }
    }
    sum = min(sum, 1);

    output[y * width + x] = dicomSample * (1 - sum);
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
    simpleKernel<<<grid, block>>>(outputDev, tex, imageWidth, imageHeight);

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
    simpleKernel<<<grid, block>>>(outputDev, tex, imageWidth, imageHeight);

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

__host__ short* modifyTextureRasterized(int imageWidth, int imageHeight, short* dicomData, unsigned char* polylineData)
{
    cudaError_t e;
    size_t sizeDicom = imageWidth * imageHeight * sizeof(short);
    size_t sizePolyline = imageWidth * imageHeight * sizeof(unsigned char);

    // Allocate a cudaArray in device memory and copy our dicomData there
    cudaChannelFormatDesc channelDescD = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaArray* cuArrayD;
    cudaMallocArray(&cuArrayD, &channelDescD, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArrayD, 0, 0, dicomData, sizeDicom, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescD;
    memset(&resDescD, 0, sizeof(resDescD));
    resDescD.resType = cudaResourceTypeArray;
    resDescD.res.array.array = cuArrayD;

    cudaTextureDesc texDescD;
    memset(&texDescD, 0, sizeof(texDescD));
    texDescD.readMode = cudaReadModeElementType;
    texDescD.addressMode[0] = cudaAddressModeClamp;
    texDescD.addressMode[1] = cudaAddressModeClamp;
    texDescD.filterMode = cudaFilterModePoint;
    texDescD.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t texD = 0;
    cudaCreateTextureObject(&texD, &resDescD, &texDescD, NULL);

    // Allocate a cudaArray in device memory and copy our polylineData there
    cudaChannelFormatDesc channelDescP = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* cuArrayP;
    cudaMallocArray(&cuArrayP, &channelDescP, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArrayP, 0, 0, polylineData, sizePolyline, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescP;
    memset(&resDescP, 0, sizeof(resDescP));
    resDescP.resType = cudaResourceTypeArray;
    resDescP.res.array.array = cuArrayP;

    cudaTextureDesc texDescP;
    memset(&texDescP, 0, sizeof(texDescP));
    texDescP.readMode = cudaReadModeElementType;
    texDescP.addressMode[0] = cudaAddressModeClamp;
    texDescP.addressMode[1] = cudaAddressModeClamp;
    texDescP.filterMode = cudaFilterModePoint;
    texDescP.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t texP = 0;
    cudaCreateTextureObject(&texP, &resDescP, &texDescP, NULL);

    // Create result array in the device
    short* outputDev;
    cudaMalloc(&outputDev, sizeDicom);

    // Create result array in the host
    short* outputHost = (short*)malloc(sizeDicom);

    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    rasterizerTest<<<grid, block>>>(outputDev, texD, texP, imageWidth, imageHeight);

    // Copy results to host
    cudaMemcpy(outputHost, outputDev, sizeDicom, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(texD);
    cudaDestroyTextureObject(texP);
    cudaFreeArray(cuArrayD);
    cudaFreeArray(cuArrayP);
    cudaFree(outputDev);

    return outputHost;
}

__host__ unsigned short* modifyTextureRasterized(int imageWidth, int imageHeight, unsigned short* dicomData, unsigned char* polylineData)
{
    cudaError_t e;
    size_t sizeDicom = imageWidth * imageHeight * sizeof(unsigned short);
    size_t sizePolyline = imageWidth * imageHeight * sizeof(unsigned char);

    // Allocate a cudaArray in device memory and copy our dicomData there
    cudaChannelFormatDesc channelDescD = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* cuArrayD;
    cudaMallocArray(&cuArrayD, &channelDescD, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArrayD, 0, 0, dicomData, sizeDicom, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescD;
    memset(&resDescD, 0, sizeof(resDescD));
    resDescD.resType = cudaResourceTypeArray;
    resDescD.res.array.array = cuArrayD;

    cudaTextureDesc texDescD;
    memset(&texDescD, 0, sizeof(texDescD));
    texDescD.readMode = cudaReadModeElementType;
    texDescD.addressMode[0] = cudaAddressModeClamp;
    texDescD.addressMode[1] = cudaAddressModeClamp;
    texDescD.filterMode = cudaFilterModePoint;
    texDescD.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t texD = 0;
    cudaCreateTextureObject(&texD, &resDescD, &texDescD, NULL);

    // Allocate a cudaArray in device memory and copy our polylineData there
    cudaChannelFormatDesc channelDescP = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* cuArrayP;
    cudaMallocArray(&cuArrayP, &channelDescP, imageWidth, imageHeight);
    cudaMemcpyToArray(cuArrayP, 0, 0, polylineData, sizePolyline, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescP;
    memset(&resDescP, 0, sizeof(resDescP));
    resDescP.resType = cudaResourceTypeArray;
    resDescP.res.array.array = cuArrayP;

    cudaTextureDesc texDescP;
    memset(&texDescP, 0, sizeof(texDescP));
    texDescP.readMode = cudaReadModeElementType;
    texDescP.addressMode[0] = cudaAddressModeClamp;
    texDescP.addressMode[1] = cudaAddressModeClamp;
    texDescP.filterMode = cudaFilterModePoint;
    texDescP.normalizedCoords = 1;

    // Create bindless texture object
    cudaTextureObject_t texP = 0;
    cudaCreateTextureObject(&texP, &resDescP, &texDescP, NULL);

    // Create result array in the device
    unsigned short* outputDev;
    cudaMalloc(&outputDev, sizeDicom);

    // Create result array in the host
    unsigned short* outputHost = (unsigned short*)malloc(sizeDicom);

    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    rasterizerTest<<<grid, block>>>(outputDev, texD, texP, imageWidth, imageHeight);

    // Copy results to host
    cudaMemcpy(outputHost, outputDev, sizeDicom, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(texD);
    cudaDestroyTextureObject(texP);
    cudaFreeArray(cuArrayD);
    cudaFreeArray(cuArrayP);
    cudaFree(outputDev);

    return outputHost;
}
