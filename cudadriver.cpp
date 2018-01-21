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

#include "cudadriver.h"
#include "kernels.cuh"
#include "gradient.cuh"
#include "gaussian.cuh"
#include "common.cuh"

//Makes these long declarations a little more readable
#define VTK_NEW(type, instance); vtkSmartPointer<type> instance = vtkSmartPointer<type>::New();



vtkSmartPointer<vtkImageData> testSobelFilter(vtkImageData* input)
{
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

    auto outInfo = outputImage->GetInformation();
    vtkImageData::SetScalarType(VTK_FLOAT, outInfo);
    vtkImageData::SetNumberOfScalarComponents(3, outInfo);
    outputImage->AllocateScalars(outInfo);

    //Get pointer to input data
    float* inputScalarPointer = static_cast<float*>(input->GetScalarPointer());

    //Execute kernel
    float* outputScalarPointer = applySobelFilter(inputDims[0], inputDims[1], inputScalarPointer);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    // Split interlaced channels into separate RGB channels
    unsigned int width = inputDims[1];
    unsigned int numPixels = inputDims[0] * width;
    for(unsigned int i = 0; i < numPixels; i++)
    {
        auto red   = outputScalarPointer[i*2];
        auto green = outputScalarPointer[i*2 + 1];
        auto blue  = 0.0f;

        outputImage->SetScalarComponentFromFloat(i % width, i / width, 0, 0, std::abs(red));
        outputImage->SetScalarComponentFromFloat(i % width, i / width, 0, 1, std::abs(green));
        outputImage->SetScalarComponentFromFloat(i % width, i / width, 0, 2, std::abs(blue));
    }
    free(outputScalarPointer);

    return outputImage;
}

vtkSmartPointer<vtkImageData> testGaussianFilter(vtkImageData* input)
{
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

    //Get pointer to input data
    float* inputScalarPointer = static_cast<float*>(input->GetScalarPointer());

    //Execute kernel
    float* outputScalarPointer = applyGaussianFilter(inputDims[0], inputDims[1], inputScalarPointer);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    //Insert results into a vtk array
    VTK_NEW(vtkFloatArray, outputArr);
    outputArr->SetNumberOfComponents(1);
    outputArr->SetArray(outputScalarPointer, inputDims[0] * inputDims[1], 1); //last 1 to take ownership of outputScalarPointer, will use its memory

    //Insert vtk array into result image
    auto outInfo = outputImage->GetInformation();
    vtkImageData::SetScalarType(VTK_FLOAT, outInfo);
    vtkImageData::SetNumberOfScalarComponents(1, outInfo);
    outputImage->SetInformation(outInfo);
    outputImage->GetPointData()->SetScalars(outputArr);

    return outputImage;
}

vtkSmartPointer<vtkImageData> testEdgeIndicator(vtkImageData* input)
{
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

    //Get pointer to input data
    float* inputScalarPointer = static_cast<float*>(input->GetScalarPointer());

    //Execute kernel
    float* outputScalarPointer = applyEdgeIndicator(inputDims[0], inputDims[1], inputScalarPointer);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return nullptr;
    }

    //Insert results into a vtk array
    VTK_NEW(vtkFloatArray, outputArr);
    outputArr->SetNumberOfComponents(1);
    outputArr->SetArray(outputScalarPointer, inputDims[0] * inputDims[1], 1); //last 1 to take ownership of outputScalarPointer, will use its memory

    //Insert vtk array into result image
    auto outInfo = outputImage->GetInformation();
    vtkImageData::SetScalarType(VTK_FLOAT, outInfo);
    vtkImageData::SetNumberOfScalarComponents(1, outInfo);
    outputImage->SetInformation(outInfo);
    outputImage->GetPointData()->SetScalars(outputArr);

    return outputImage;
}

void initLevelSets(vtkImageData* dicomInput, vtkImageData* polylineInput)
{
    int* inputDims = dicomInput->GetDimensions();

    //Copy input spatial info into a result image
    VTK_NEW(vtkImageData, outputImage);
    double spacing[3];
    int dim[3];
    double origin[3];
    int extent[6];
    dicomInput->GetSpacing(spacing);
    dicomInput->GetDimensions(dim);
    dicomInput->GetOrigin(origin);
    dicomInput->GetExtent(extent);
    outputImage->SetDimensions(dim);
    outputImage->SetSpacing(spacing);
    outputImage->SetOrigin(origin);
    outputImage->SetExtent(extent);

    //Get pointer to input data
    float* h_dicomData = static_cast<float*>(dicomInput->GetScalarPointer());
    float* h_polylineData = static_cast<float*>(polylineInput->GetScalarPointer());

    LevelSetData lsd;
    initLevelSetData(inputDims[0], inputDims[1], h_dicomData, h_polylineData, &lsd);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

vtkSmartPointer<vtkImageData> iterateLevelSets(unsigned int numIters)
{
    return nullptr;
}
