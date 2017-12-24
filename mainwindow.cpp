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
    tracer->SetProjectToPlane(1);
    tracer->SetProjectionNormalToZAxes();
    tracer->SetProjectionPosition(1);
    tracer->SetAutoClose(1);
    tracer->On();

    //Keep track of this polyline (easiest way of cleaning it later)
    m_polylines.push_back(tracer.GetPointer());

    this->ui->qvtkWidget->repaint();
    interactor->Start();
}

void MainWindow::on_actionClear_polylines_triggered()
{
    for(int i = 0; i < m_polylines.size(); i++)
    {
        vtkImageTracerWidget* widget = m_polylines[i];
        widget->Off();
        widget->Delete();
    }
    m_polylines.clear();
    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionRasterize_polylines_triggered()
{
    VTK_NEW(vtkSphereSource, sphereSource);
    sphereSource->SetRadius(20);
    sphereSource->SetPhiResolution(30);
    sphereSource->SetThetaResolution(30);
    sphereSource->Update();
    vtkSmartPointer<vtkPolyData> pd1 = sphereSource->GetOutput();

    VTK_NEW(vtkLineSource, lineSource);
    lineSource->SetPoint1(0, 0, 0);
    lineSource->SetPoint2(100, 100, 0);
    lineSource->Update();
    vtkSmartPointer<vtkPolyData> pd2 = lineSource->GetOutput();

    VTK_NEW(vtkLinearExtrusionFilter, linearFilterX);
    linearFilterX->SetInputData(pd2);
    linearFilterX->SetCapping(1);
    linearFilterX->SetScaleFactor(1);
    linearFilterX->SetExtrusionTypeToNormalExtrusion();
    linearFilterX->SetVector(1, 0, 0);
    linearFilterX->Update();

    VTK_NEW(vtkLinearExtrusionFilter, linearFilterY);
    linearFilterY->SetInputData(linearFilterX->GetOutput());
    linearFilterY->SetCapping(1);
    linearFilterY->SetScaleFactor(1);
    linearFilterY->SetExtrusionTypeToNormalExtrusion();
    linearFilterY->SetVector(0, 1, 1);
    linearFilterY->Update();

    VTK_NEW(vtkLinearExtrusionFilter, linearFilterZ);
    linearFilterZ->SetInputData(linearFilterY->GetOutput());
    linearFilterZ->SetCapping(1);
    linearFilterZ->SetScaleFactor(1);
    linearFilterZ->SetExtrusionTypeToNormalExtrusion();
    linearFilterZ->SetVector(0, 0, 1);
    linearFilterZ->Update();

    vtkSmartPointer<vtkPolyData> pd3 = linearFilterZ->GetOutput();

    VTK_NEW(vtkAppendPolyData, appendFilter);
    appendFilter->AddInputData(pd1);
    appendFilter->AddInputData(pd3);
    appendFilter->Update();
    vtkSmartPointer<vtkPolyData> pd = appendFilter->GetOutput();

    VTK_NEW(vtkPolyDataMapper, mapper);
    mapper->SetInputData(pd);

    VTK_NEW(vtkActor, polyActor);
    polyActor->SetMapper(mapper);

    VTK_NEW(vtkImageData, whiteImage);
    double bounds[6];
    pd->GetBounds(bounds);
    double spacing[3]; // desired volume spacing
    spacing[0] = 0.5;
    spacing[1] = 0.5;
    spacing[2] = 0.5;
    whiteImage->SetSpacing(spacing);

    // compute dimensions
    int dim[3];
    for (int i = 0; i < 3; i++)
    {
        dim[i] = static_cast<int>(ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]));
    }
    whiteImage->SetDimensions(dim);
    whiteImage->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, 0);

    double origin[3];
    origin[0] = bounds[0] + spacing[0] / 2;
    origin[1] = bounds[2] + spacing[1] / 2;
    origin[2] = 0;
    whiteImage->SetOrigin(origin);
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
    pol2stenc->SetInputData(pd);
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

    m_renderer->RemoveAllViewProps();
    m_renderer->AddActor(imgActor);
    //m_renderer->AddActor(polyActor);
    m_renderer->ResetCamera();

    this->ui->qvtkWidget->repaint();
}
