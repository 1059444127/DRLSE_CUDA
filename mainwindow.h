#include <QMainWindow>

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

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

    private slots:
    void on_actionOpenFile_triggered();
    void on_actionExit_triggered();

    private:
    //Pointer to the class inside the Ui namespace
    Ui::MainWindow *ui;

    //Rendereres
    vtkSmartPointer<vtkRenderer> m_renderer;
};
