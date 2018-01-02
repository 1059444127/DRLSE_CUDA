#include <vector>

#include <QMainWindow>

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkPolyLineWidget.h>
#include <vtkImageTracerWidget.h>
#include <vtkPropPicker.h>
#include <vtkCornerAnnotation.h>



// Template for image value reading
template<typename T>
void vtkValueMessageTemplate(vtkImageData* image, int* position,
                             std::string& message)
{
  T* tuple = ((T*)image->GetScalarPointer(position));
  int components = image->GetNumberOfScalarComponents();
  for (int c = 0; c < components; ++c)
  {
    message += to_string(tuple[c]);
    if (c != (components - 1))
    {
      message += ", ";
    }
  }
}




namespace Ui{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    //Special macro. Adds some functionality on top of C++ classes,
    //like the ability to query class and slot names during run-time
    //Also possible to query a slot's parameter types and invoke it
    Q_OBJECT

    public:
    MainWindow();
    ~MainWindow(){};

    void closeEvent(QCloseEvent *event) override;

    static MainWindow* instance;

    void ShowStatus(std::string message);

    vtkImageActor* GetActor() {return m_mainActor.GetPointer();}
    vtkRenderer* GetRenderer() {return m_renderer.GetPointer();}
    vtkPropPicker* GetPicker() {return m_picker.GetPointer();}
    vtkCornerAnnotation* GetCornerAnnotation() {return m_cornerAnn.GetPointer();}

    private slots:
    void on_actionOpenFile_triggered();
    void on_actionOpen_Folder_triggered();
    void on_actionExit_triggered();
    void on_actionReset_view_triggered();    
    void on_actionCreate_polyline_triggered();
    void on_actionClear_polylines_triggered();
    void on_actionRasterize_polylines_triggered();
    void on_actionTest_Sobel_filter_triggered();
    void on_actionTest_Gaussian_filter_triggered();
    void on_actionTest_edge_indicator_triggered();

private:
    Ui::MainWindow *ui;

    vtkSmartPointer<vtkImageActor> m_mainActor;
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkPropPicker> m_picker;
    vtkSmartPointer<vtkCornerAnnotation> m_cornerAnn;

    std::vector<vtkSmartPointer<vtkImageTracerWidget>> m_polylines;
    vtkSmartPointer<vtkImageActor> m_polyLineActor;
};
