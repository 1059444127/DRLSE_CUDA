#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include <cuda_runtime.h>

vtkSmartPointer<vtkImageData> testSobelFilter(vtkImageData* input);
