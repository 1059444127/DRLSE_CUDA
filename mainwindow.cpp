#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <assert.h>

#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>

#include <vtkRenderWindow.h>
#include <vtkImageMapper3D.h>
#include <vtkImageActor.h>
#include <vtkCamera.h>
#include <vtkDICOMImageReader.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLineWidget.h>
#include <vtkPolyLineRepresentation.h>
#include <vtkPoints.h>
#include <vtkPropCollection.h>
#include <vtkProp.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkImageStencil.h>
#include <vtkPointData.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include <vtkSphereSource.h>
#include <vtkRotationalExtrusionFilter.h>

#include <vtkImageTracerWidget.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkProperty.h>
#include <vtkCallbackCommand.h>
#include <vtkImageProperty.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkTubeFilter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cudadriver.h"

//Makes these long declarations a little more readable
#define VTK_NEW(type, instance); vtkSmartPointer<type> instance = vtkSmartPointer<type>::New();

MainWindow::MainWindow()
{
    //Configure the ui
    this->ui = new Ui::MainWindow;
    this->ui->setupUi(this);

    //Update status bar
    this->statusBar()->showMessage("Ready");
    QApplication::processEvents();

    //Add the renderer to the render window Qt widget
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    this->ui->qvtkWidget->GetRenderWindow()->AddRenderer(m_renderer);
}

void MainWindow::ShowStatus(std::string message)
{
    this->statusBar()->showMessage(QString::fromStdString(message));
}

void MainWindow::on_actionOpenFile_triggered()
{
    //getOpenFileName displays a file dialog and returns the full file path of the selected file, or an empty string if the user canceled the dialog
    //The tr() function makes the dialog language proof (chinese characters, etc)
    QString fileName = QFileDialog::getOpenFileName(this, tr("Pick a DICOM file"), QString(), tr("All files (*.*);;DICOM FILES (*.dcm)"));

    if(!fileName.isEmpty())
    {
        std::string fileNameStd = fileName.toStdString();

        //Read all DICOM files in the specified directory
        VTK_NEW(vtkDICOMImageReader, dicomReader);
        dicomReader->SetFileName(fileNameStd.c_str());
        dicomReader->Update();
        auto dicomImage = dicomReader->GetOutput();

        m_mainActor = vtkSmartPointer<vtkImageActor>::New();
        m_mainActor->GetMapper()->SetBackground(125);
        m_mainActor->GetMapper()->SetInputData(dicomImage);
        //m_mainActor->SetOpacity(0.5);

        VTK_NEW(vtkRenderWindowInteractor, interactor);
        interactor->SetRenderWindow(m_renderer->GetRenderWindow());

        VTK_NEW(vtkInteractorStyleImage, style);
        interactor->SetInteractorStyle(style);

        //m_renderer->AddActor(canvasImageActor);
        m_renderer->AddActor(m_mainActor);
        m_renderer->GetActiveCamera()->SetParallelProjection(1);
        m_renderer->ResetCamera();

        interactor->Start();
    }
}

void MainWindow::on_actionOpen_Folder_triggered()
{
    //getOpenFileName displays a file dialog and returns the full file path of the selected file, or an empty string if the user canceled the dialog
    //The tr() function makes the dialog language proof (chinese characters, etc)
    //QString fileName = QFileDialog::getOpenFileName(this, tr("Pick a DICOM file"), QString(), tr("Text Files (*.txt)"));

    //QString folderName = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

//    if(!folderName.isEmpty())
//    {
//        //Read all DICOM files in the specified directory
//        VTK_NEW(vtkDICOMImageReader, dicomReader);
//        dicomReader->SetFileN

        //Create an input file stream
//        std::ifstream inputFile(fileName.toStdString().c_str());

//        //If the file isn't open, show an error box
//        if(!inputFile)
//        {
//            QMessageBox::critical(this, tr("Error"), tr("Could not open file"));
//            return;
//        }

//        //Get the file size
//        inputFile.seekg(0, std::ios::end);
//        int fileSize = inputFile.tellg();
//        inputFile.seekg(0, std::ios::beg);

//        //Read the file into data
//        std::vector<uint8_t> data(fileSize);
//        inputFile.read((char*) &data[0], fileSize);

//        //Extract the data dimensions from the header
//        std::memcpy(&m_rawLength, &(data[16]), 4*sizeof(uint8_t));
//        std::memcpy(&m_rawWidth, &(data[20]), 4*sizeof(uint8_t));
//        std::memcpy(&m_rawDepth, &(data[24]), 4*sizeof(uint8_t));

//        //Setup to display oct data with 0 bytes of frameHeaders
//        this->setUpForOctRaw(m_rawOctPolyData, data, 0, 512);

//        //Displays the file name in the ui
//        this->ui->FileBox->setText(fileName);

//        //Set the state
//        m_newRawData = true;

//        //Changes the visualization and control menu to the raw data items
//        this->ui->comboBox->setCurrentIndex(0);
//        this->ui->controlPanel->setCurrentIndex(0);

//        //Renders the oct data
//        this->renderOctRaw();
//    }
}

void MainWindow::on_actionExit_triggered()
{
    qApp->exit();
}

void MainWindow::on_actionReset_view_triggered()
{
    m_renderer->ResetCamera();
    m_renderer->GetActiveCamera()->SetViewUp(0,1,0);
    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionTest_CUDA_triggered()
{
    //Get currently displayed imageData
    auto inputData = m_mainActor->GetInput();

    //Apply our CUDA kernel to it
    auto outputData = runCuda(inputData);

    //Display results
    m_mainActor->GetMapper()->RemoveAllInputs();
    m_mainActor->SetInputData(outputData);
    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionCreate_polyline_triggered()
{
    //Stop manipulating other polylines
    for(int i = 0; i < m_polylines.size(); i++)
    {
        vtkImageTracerWidget* widget = m_polylines[i];
        widget->InteractionOff();
    }

    auto interactor = m_renderer->GetRenderWindow()->GetInteractor();

    VTK_NEW(vtkImageTracerWidget, tracer);
    tracer->GetLineProperty()->SetLineWidth(2);
    tracer->SetInteractor(interactor);
    tracer->SetViewProp(m_mainActor);
    tracer->SetProjectToPlane(1); //place widget lines 1 unit in front of the main image plane
    tracer->SetProjectionNormalToZAxes();
    tracer->SetProjectionPosition(1);
    tracer->SetAutoClose(1);
    tracer->On();

    //Keep track of this polyline (easiest way of cleaning it later)
    m_polylines.push_back(tracer);

    this->ui->qvtkWidget->repaint();
    interactor->Start();
}

void MainWindow::on_actionClear_polylines_triggered()
{
    for(int i = 0; i < m_polylines.size(); i++)
    {
        auto widget = m_polylines[i];
        widget->Off();
    }
    m_polylines.clear();
    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionRasterize_polylines_triggered()
{
    VTK_NEW(vtkAppendPolyData, appendFilter);
    for(int i = 0; i < m_polylines.size(); i++)
    {
        auto widget = m_polylines[i];

        VTK_NEW(vtkPolyData, pd);
        widget->GetPath(pd);
        appendFilter->AddInputData(pd);
    }
    appendFilter->Update();

    //Translate polydata down in the Z direction so it sits on top of the
    //main image. Its usually 1 unit ahead of it so we can see it on top of the image
    VTK_NEW(vtkTransform, trans);
    trans->Translate(0, 0, -1);

    VTK_NEW(vtkTransformPolyDataFilter, transFilter);
    transFilter->SetInputData(appendFilter->GetOutput());
    transFilter->SetTransform(trans);
    transFilter->Update();

    VTK_NEW(vtkTubeFilter, tube);
    tube->SetInputData(transFilter->GetOutput());
    tube->SetRadius(0.3);
    tube->SetNumberOfSides(6);
    tube->CappingOn();
    tube->Update();

    VTK_NEW(vtkPolyDataMapper, mapper);
    mapper->SetInputData(tube->GetOutput());

    VTK_NEW(vtkActor, polyActor);
    polyActor->SetMapper(mapper);

    VTK_NEW(vtkImageData, whiteImage);
    double spacing[3];
    int dim[3];
    double origin[3];
    int extent[6];
    auto mainImage = m_mainActor->GetInput();
    mainImage->GetSpacing(spacing);
    mainImage->GetDimensions(dim);
    mainImage->GetOrigin(origin);
    mainImage->GetExtent(extent);

    whiteImage->SetDimensions(dim);
    whiteImage->SetSpacing(spacing);
    whiteImage->SetOrigin(origin);
    whiteImage->SetExtent(extent);
    whiteImage->AllocateScalars(VTK_UNSIGNED_CHAR,1);

    // fill the image with foreground voxels:
    unsigned char inval = 255;
    unsigned char outval = 0;
    vtkIdType count = whiteImage->GetNumberOfPoints();
    for (vtkIdType i = 0; i < count; ++i)
    {
        whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
    }

    // polygonal data --> image stencil:
    VTK_NEW(vtkPolyDataToImageStencil, pol2stenc);
    pol2stenc->SetInputData(tube->GetOutput());
    pol2stenc->SetOutputOrigin(origin);
    pol2stenc->SetOutputSpacing(spacing);
    pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
    pol2stenc->Update();

    // cut the corresponding white image and set the background:
    VTK_NEW(vtkImageStencil, imgstenc);
    imgstenc->SetInputData(whiteImage);
    imgstenc->SetStencilConnection(pol2stenc->GetOutputPort());
    imgstenc->ReverseStencilOff();
    imgstenc->SetBackgroundValue(outval);
    imgstenc->Update();

    VTK_NEW(vtkImageActor, imgActor);
    imgActor->SetInputData(imgstenc->GetOutput());
    imgActor->GetProperty()->SetInterpolationTypeToNearest();
    imgActor->GetMapper()->BackgroundOff();
    imgActor->SetOpacity(0.5);

    m_mainActor->GetInput()->Print(std::cout);
    whiteImage->Print(std::cout);

    //m_renderer->RemoveAllViewProps();
    m_renderer->AddActor(imgActor);
    //m_renderer->AddActor(polyActor);
    //m_renderer->ResetCamera();

    this->ui->qvtkWidget->repaint();

    on_actionClear_polylines_triggered();
}
