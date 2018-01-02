#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <kernels.cuh>
#include <common.cuh>

//====================================================================================
//KERNELS
//====================================================================================
template<typename T>
__global__ void simpleKernel(T* output, cudaTextureObject_t input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    output[y * width + x] = (T)(0.5 * tex2D<T>(input, u, v));
}

template<typename T>
__global__ void rasterizerTest(T* output, cudaTextureObject_t dicomTex, cudaTextureObject_t polylineTex, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Convert to texture units
    float u = x / (float)width;
    float v = y / (float)height;

    unsigned int sum = 0;

    //Set sum to 1 if we're next to a painted square on polylineTex
    T dicomSample = tex2D<T>(dicomTex, u, v);
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

__global__ void rasterizerTestSurface(cudaSurfaceObject_t input, cudaSurfaceObject_t output, cudaSurfaceObject_t polyline, int width, int height)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 1 && x < width-2 && y > 1 && y < height-2)
    {
        int sum = 0;
        short sample;
        for(int i = -2; i <= 2; i++)
        {
            for(int j = -2; j <= 2; j++)
            {
                surf2Dread(&sample, input, (int)((x+i)*sizeof(sample)), (int)(y+j));
                //sum += sample * d_gaussKernel5[5*i+j];
            }
        }

        surf2Dwrite((short)sum, output, x * sizeof(sample), y);
    }
}

//====================================================================================
//HOST CUDA FUNCTIONS
//====================================================================================
template<typename T, cudaChannelFormatKind FK>
__host__ T* modifyTexture(int imageWidth, int imageHeight, T* textureData)
{
    size_t size = imageWidth * imageHeight * sizeof(T);

    // Allocate a cudaArray in device memory and copy our texture data there
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8 * sizeof(T), 0, 0, 0, FK);
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
    T* outputDev;
    cudaMalloc(&outputDev, size);

    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    simpleKernel<<<grid, block>>>(outputDev, tex, imageWidth, imageHeight);

    // Create result array in the host
    T* outputHost = (T*)malloc(size);

    // Copy results to host
    cudaMemcpy(outputHost, outputDev, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cuArray);
    cudaFree(outputDev);

    return outputHost;
}

template<typename T, cudaChannelFormatKind FK>
__host__ T* modifyTextureRasterized(int imageWidth, int imageHeight, T* dicomData, unsigned char* polylineData)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(T);
    size_t sizePolyline = imageWidth * imageHeight * sizeof(unsigned char);

    // Allocate a cudaArray in device memory and copy our dicomData there
    cudaChannelFormatDesc channelDescD = cudaCreateChannelDesc(8 * sizeof(T), 0, 0, 0, FK);
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
    T* outputDev;
    cudaMalloc(&outputDev, sizeDicom);

    // Create result array in the host
    T* outputHost = (T*)malloc(sizeDicom);

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

template<typename T, cudaChannelFormatKind FK>
__host__ T* modifySurfaceRasterized(int imageWidth, int imageHeight, T* h_dataDicom, unsigned char* h_dataPolyline)
{
    size_t sizeDicom = imageWidth * imageHeight * sizeof(T);
    size_t sizePolyline = imageWidth * imageHeight * sizeof(unsigned char);

    // Create a Surface with our image data and copy that data to the device
    cudaChannelFormatDesc channelFormatDicom = cudaCreateChannelDesc(8 * sizeof(T), 0, 0, 0, FK);
    cudaArray* d_arrayDicom;
    cudaMallocArray(&d_arrayDicom, &channelFormatDicom, imageWidth, imageHeight);
    cudaMemcpyToArray(d_arrayDicom, 0, 0, h_dataDicom, sizeDicom, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescDicom;
    memset(&resDescDicom, 0, sizeof(resDescDicom));
    resDescDicom.resType = cudaResourceTypeArray;
    resDescDicom.res.array.array = d_arrayDicom;

    cudaSurfaceObject_t d_surfDicom = 0;
    cudaCreateSurfaceObject(&d_surfDicom, &resDescDicom);


    // Create a surface with our polyline data and copy that data to the device
    cudaChannelFormatDesc channelFormatPolyline = cudaCreateChannelDesc(8 * 1, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* d_arrayPolyline;
    cudaMallocArray(&d_arrayPolyline, &channelFormatPolyline, imageWidth, imageHeight);
    cudaMemcpyToArray(d_arrayPolyline, 0, 0, h_dataPolyline, sizePolyline, cudaMemcpyHostToDevice);

    cudaResourceDesc resDescPolyline;
    memset(&resDescPolyline, 0, sizeof(resDescPolyline));
    resDescPolyline.resType = cudaResourceTypeArray;
    resDescPolyline.res.array.array = d_arrayPolyline;

    cudaSurfaceObject_t d_surfPolyline = 0;
    cudaCreateSurfaceObject(&d_surfPolyline, &resDescPolyline);


    // Create an output surface
    cudaArray* d_arrayResult;
    cudaMallocArray(&d_arrayResult, &channelFormatDicom, imageWidth, imageHeight);

    cudaResourceDesc resDescResult;
    memset(&resDescResult, 0, sizeof(resDescResult));
    resDescResult.resType = cudaResourceTypeArray;
    resDescResult.res.array.array = d_arrayResult;

    cudaSurfaceObject_t d_surfResult = 0;
    cudaCreateSurfaceObject(&d_surfResult, &resDescResult);


    // Run kernel
    dim3 block(imageWidth / 16, imageHeight / 16,1);
    dim3 grid(16,16,1);
    rasterizerTestSurface<<<grid, block>>>(d_surfDicom, d_surfResult, d_surfPolyline, imageWidth, imageHeight);

    // Copy results to host
    T* outputHost = (T*)malloc(sizeDicom);
    cudaMemcpyFromArray(outputHost, d_arrayResult, 0, 0, sizeDicom, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroySurfaceObject(d_surfDicom);
    cudaDestroySurfaceObject(d_surfPolyline);
    cudaDestroySurfaceObject(d_surfResult);
    cudaFreeArray(d_arrayDicom);
    cudaFreeArray(d_arrayPolyline);
    cudaFreeArray(d_arrayResult);

    return outputHost;
}

//====================================================================================
//TEMPLATE EXPLICIT INSTANTIATIONS
//====================================================================================
//Explicit instantiation so these signatures are available when the linker is linking our lib to the exe
//The compiler would have no way of knowing these specific signatures will be needed since the calls are in
//another compilation unit
template __host__ short* modifyTexture<short, cudaChannelFormatKindSigned>(int imageWidth, int imageHeight, short* textureData);
template __host__ unsigned short* modifyTexture<unsigned short, cudaChannelFormatKindUnsigned>(int imageWidth, int imageHeight, unsigned short* textureData);
template __host__ short* modifyTextureRasterized<short, cudaChannelFormatKindSigned>(int imageWidth, int imageHeight, short* dicomData, unsigned char* polylineData);
template __host__ unsigned short* modifyTextureRasterized<unsigned short, cudaChannelFormatKindUnsigned>(int imageWidth, int imageHeight, unsigned short* dicomData, unsigned char* polylineData);
template __host__ short* modifySurfaceRasterized<short, cudaChannelFormatKindSigned>(int imageWidth, int imageHeight, short* dicomData, unsigned char* polylineData);
template __host__ unsigned short* modifySurfaceRasterized<unsigned short, cudaChannelFormatKindUnsigned>(int imageWidth, int imageHeight, unsigned short* dicomData, unsigned char* polylineData);
