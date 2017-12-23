#include <vector>

#include <QMainWindow>

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkPolyLineWidget.h>

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
    void ShowStatus(std::string message);

    private slots:
    void on_actionOpenFile_triggered();
    void on_actionOpen_Folder_triggered();
    void on_actionExit_triggered();
    void on_actionReset_view_triggered();
    void on_actionTest_CUDA_triggered();
    void on_actionCreate_polyline_triggered();
    void on_actionClear_polylines_triggered();

private:
    Ui::MainWindow *ui;

    vtkSmartPointer<vtkImageActor> m_mainActor;
    vtkSmartPointer<vtkRenderer> m_renderer;
    std::vector<vtkPolyLineWidget*> m_polylines;
};
