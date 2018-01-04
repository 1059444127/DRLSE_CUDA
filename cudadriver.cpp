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
#include <gaussian.cuh>

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
