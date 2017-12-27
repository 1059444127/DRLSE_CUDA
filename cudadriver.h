#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include <cuda_runtime.h>

vtkSmartPointer<vtkImageData> testCUDA(vtkImageData* input);
vtkSmartPointer<vtkImageData> testCUDAandRasterized(vtkImageData* dicomImageData, vtkImageData* polyLineImageData);
vtkSmartPointer<vtkImageData> testConvolution(vtkImageData* input);


