#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include "common.cuh"

class CUDADriver
{
public:

    static CUDADriver* instance;

    vtkSmartPointer<vtkImageData> testSobelFilter(vtkImageData* input);
    vtkSmartPointer<vtkImageData> testGaussianFilter(vtkImageData* input);
    vtkSmartPointer<vtkImageData> testEdgeIndicator(vtkImageData* input);
    void initLevelSets(vtkImageData* imageInput, vtkImageData* polylineInput);
    vtkSmartPointer<vtkImageData> iterateLevelSets(unsigned int numIters);

private:
    std::unique_ptr<LevelSetData> m_testLSD;
};



