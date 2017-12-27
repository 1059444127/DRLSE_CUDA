#include <stdio.h>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkUnsignedShortArray.h>
#include <vtkShortArray.h>
#include <vtkAbstractArray.h>
#include <vtkDataArray.h>

#include <cuda_runtime.h>

#include <cudadriver.h>
#include <kernels.cuh>

//Makes these long declarations a little more readable
#define VTK_NEW(type, instance); vtkSmartPointer<type> instance = vtkSmartPointer<type>::New();

vtkSmartPointer<vtkImageData> testCUDA(vtkImageData* input)
{
    auto type = std::string(input->GetScalarTypeAsString());
    int* inputDims = input->GetDimensions();
    VTK_NEW(vtkImageData, outputImage);

    if(type == "short")
    {
        short* inputScalarPointer = static_cast<short*>(input->GetScalarPointer());
        short* outputScalarPointer =
                modifyTexture<short, cudaChannelFormatKindSigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

        VTK_NEW(vtkShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((short*)outputScalarPointer,
                            4, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(input);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else if(type == "unsigned short")
    {
        unsigned short* inputScalarPointer = static_cast<unsigned short*>(input->GetScalarPointer());
        unsigned short* outputScalarPointer =
                modifyTexture<unsigned short, cudaChannelFormatKindUnsigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

        VTK_NEW(vtkUnsignedShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((unsigned short*)outputScalarPointer,
                            5, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(input);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else
    {
        fprintf(stderr, "Invalid vtkImageData scalar type!");
        return nullptr;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    return outputImage;
}

vtkSmartPointer<vtkImageData> testCUDAandRasterized(vtkImageData* dicomImageData, vtkImageData* polyLineImageData)
{
    unsigned char* polyLineScalarPointer = static_cast<unsigned char*>(polyLineImageData->GetScalarPointer());

    auto type = std::string(dicomImageData->GetScalarTypeAsString());
    int* inputDims = dicomImageData->GetDimensions();
    VTK_NEW(vtkImageData, outputImage);

    if(type == "short")
    {
        short* dicomScalarPointer = static_cast<short*>(dicomImageData->GetScalarPointer());
        short* outputScalarPointer =
                modifySurfaceRasterized<short, cudaChannelFormatKindSigned>(
                    inputDims[0],
                    inputDims[1],
                    dicomScalarPointer,
                    polyLineScalarPointer);

        VTK_NEW(vtkShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((short*)outputScalarPointer,
                            4, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(dicomImageData);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else if(type == "unsigned short")
    {
        unsigned short* dicomScalarPointer = static_cast<unsigned short*>(dicomImageData->GetScalarPointer());
        unsigned short* outputScalarPointer =
                modifySurfaceRasterized<unsigned short, cudaChannelFormatKindUnsigned>(
                    inputDims[0],
                    inputDims[1],
                    dicomScalarPointer,
                    polyLineScalarPointer);

        VTK_NEW(vtkUnsignedShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((unsigned short*)outputScalarPointer,
                            5, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(dicomImageData);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else
    {
        fprintf(stderr, "Invalid vtkImageData scalar type!");
        return nullptr;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    return outputImage;
}

vtkSmartPointer<vtkImageData> testConvolution(vtkImageData* input)
{
    auto type = std::string(input->GetScalarTypeAsString());
    int* inputDims = input->GetDimensions();
    VTK_NEW(vtkImageData, outputImage);

    if(type == "short")
    {
        short* inputScalarPointer = static_cast<short*>(input->GetScalarPointer());
        short* outputScalarPointer =
                convolveImage<short, cudaChannelFormatKindSigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

        VTK_NEW(vtkShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((short*)outputScalarPointer,
                            4, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(input);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else if(type == "unsigned short")
    {
        unsigned short* inputScalarPointer = static_cast<unsigned short*>(input->GetScalarPointer());
        unsigned short* outputScalarPointer =
                convolveImage<unsigned short, cudaChannelFormatKindUnsigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

        VTK_NEW(vtkUnsignedShortArray, outputArr);
        outputArr->SetNumberOfComponents(1);
        outputArr->SetArray((unsigned short*)outputScalarPointer,
                            5, //short
                            1); //take ownership of outputScalarPointer, will use its memory

        outputImage->DeepCopy(input);
        outputImage->GetPointData()->SetScalars(outputArr);
    }
    else
    {
        fprintf(stderr, "Invalid vtkImageData scalar type!");
        return nullptr;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    return outputImage;
}
