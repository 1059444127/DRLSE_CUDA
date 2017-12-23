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

        //Gets bounds in world coordinates (real space)
        double bounds[6] = {0};
        m_mainActor->GetBounds(bounds);

        VTK_NEW(vtkRenderWindowInteractor, interactor);
        interactor->SetRenderWindow(m_renderer->GetRenderWindow());

        VTK_NEW(vtkInteractorStyleImage, style);
        interactor->SetInteractorStyle(style);

        //m_renderer->RemoveAllViewProps();
        m_renderer->AddActor(m_mainActor);
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
    auto interactor = m_renderer->GetRenderWindow()->GetInteractor();

    VTK_NEW(vtkPoints, pts);
    pts->SetNumberOfPoints(5);
    pts->SetPoint(0, 0, 0, 0);
    pts->SetPoint(1, 20, 0, 0);
    pts->SetPoint(2, 20, 20, 0);
    pts->SetPoint(3, 25, 40, 0);
    pts->SetPoint(4, 0, 0, 0);

    VTK_NEW(vtkPolyLineWidget, lineWidget);
    lineWidget->SetInteractor(interactor);
    lineWidget->On();

    m_polylines.push_back(lineWidget.GetPointer());

    auto polyLineRep = reinterpret_cast<vtkPolyLineRepresentation*>(lineWidget->GetRepresentation());
    polyLineRep->InitializeHandles(pts);

    this->ui->qvtkWidget->repaint();
    interactor->Start();
}

void MainWindow::on_actionClear_polylines_triggered()
{
    for(int i = 0; i < m_polylines.size(); i++)
    {
        vtkPolyLineWidget* widget = m_polylines[i];
        widget->Delete();
    }
    m_polylines.clear();
    this->ui->qvtkWidget->repaint();
}
