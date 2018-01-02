#include <stdio.h>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkUnsignedShortArray.h>
#include <vtkFloatArray.h>
#include <vtkShortArray.h>
#include <vtkAbstractArray.h>
#include <vtkDataArray.h>
#include <vtkInformation.h>

#include <cuda_runtime.h>

#include <cudadriver.h>
#include <kernels.cuh>
#include <gradient.cuh>

//Makes these long declarations a little more readable
#define VTK_NEW(type, instance); vtkSmartPointer<type> instance = vtkSmartPointer<type>::New();


vtkSmartPointer<vtkImageData> testSobelFilter(vtkImageData* input)
{
    auto type = std::string(input->GetScalarTypeAsString());
    int* inputDims = input->GetDimensions();

    //Copy input spatial info into a result image
    VTK_NEW(vtkImageData, outputImage);
    double spacing[3];
    int dim[3];
    double origin[3];
    int extent[6];
    input->GetSpacing(spacing);
    input->GetDimensions(dim);
    input->GetOrigin(origin);
    input->GetExtent(extent);
    outputImage->SetDimensions(dim);
    outputImage->SetSpacing(spacing);
    outputImage->SetOrigin(origin);
    outputImage->SetExtent(extent);

    float* outputScalarPointer;
    if(type == "short")
    {
        //Get pointer to input data
        short* inputScalarPointer = static_cast<short*>(input->GetScalarPointer());

        //Execute kernel
        outputScalarPointer = applySobelFilter<short, cudaChannelFormatKindSigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

    }
    else if(type == "unsigned short")
    {
        //Get pointer to input data
        unsigned short* inputScalarPointer = static_cast<unsigned short*>(input->GetScalarPointer());

        //Execute kernel
        outputScalarPointer = applySobelFilter<unsigned short, cudaChannelFormatKindUnsigned>(
                    inputDims[0],
                    inputDims[1],
                    inputScalarPointer);

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

    //Insert results into a vtk array
    VTK_NEW(vtkFloatArray, outputArr);
    outputArr->SetNumberOfComponents(2);
    outputArr->SetArray(outputScalarPointer, inputDims[0] * inputDims[1], 1); //last 1 to take ownership of outputScalarPointer, will use its memory

    //Insert vtk array into result image
    auto outInfo = outputImage->GetInformation();
    vtkImageData::SetScalarType(VTK_FLOAT, outInfo);
    vtkImageData::SetNumberOfScalarComponents(2, outInfo);
    outputImage->SetInformation(outInfo);
    outputImage->GetPointData()->SetScalars(outputArr);

    return outputImage;
}
