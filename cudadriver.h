#include <vtkImageData.h>
#include <vtkSmartPointer.h>

vtkSmartPointer<vtkImageData> testSobelFilter(vtkImageData* input);
vtkSmartPointer<vtkImageData> testGaussianFilter(vtkImageData* input);
vtkSmartPointer<vtkImageData> testEdgeIndicator(vtkImageData* input);

void initLevelSets(vtkImageData* imageInput, vtkImageData* polylineInput);
vtkSmartPointer<vtkImageData> iterateLevelSets(unsigned int numIters);


