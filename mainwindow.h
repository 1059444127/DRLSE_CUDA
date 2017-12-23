#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <assert.h>

#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkDataSetMapper.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkProperty.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPoints.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTypeUInt8Array.h>
#include <vtkLookupTable.h>
#include <vtkAxesActor.h>
#include <vtkMaskPolyData.h>
#include <vtkMaskPoints.h>
#include <vtkClipPolyData.h>
#include <vtkPlane.h>
#include <vtkBox.h>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>
#include <vtkDelaunay2D.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkDecimatePro.h>
#include <vtkPolyDataNormals.h>
#include <vtkOutlineFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkGradientFilter.h>
#include <vtkAssignAttribute.h>

#include <QDebug>
#include <QMainWindow>
#include <QFileDialog>
#include <QFile>
#include <QMessageBox>
#include <QTextStream>
#include <QStatusBar>
#include <QVTKWidget.h>

#include <cudadriver.h>

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

#endif // MAINWINDOW_H
