/* References:
 *  picker https://lorensen.github.io/VTKExamples/site/Cxx/Images/PickPixel2/
 *
 *
 *
 *
 *
 *
 * */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <assert.h>

#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>

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
#include <vtkCornerAnnotation.h>
#include <vtkTextProperty.h>
#include <vtkPropPicker.h>
#include <vtkImageTracerWidget.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkProperty.h>
#include <vtkCallbackCommand.h>
#include <vtkImageProperty.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkTubeFilter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkAssemblyNode.h>
#include <vtkAssemblyPath.h>
#include <vtkImageCast.h>
#include <vtkImageShiftScale.h>
#include <vtkContourTriangulator.h>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cudadriver.h"

//Makes these long declarations a little more readable
#define VTK_NEW(type, instance); vtkSmartPointer<type> instance = vtkSmartPointer<type>::New();

using namespace std;

MainWindow* MainWindow::instance;

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
    m_renderer->GradientBackgroundOn();
    m_renderer->SetBackground(0, 0, 0.2);
    m_renderer->SetBackground2(0.1, 0.1, 0.1);

    this->ui->qvtkWidget->GetRenderWindow()->AddRenderer(m_renderer);

    if(MainWindow::instance == nullptr)
    {
        MainWindow::instance = this;
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    try
    {
        //We need to kill the interactor or else it will keep the event loop alive
        auto inter = m_renderer->GetRenderWindow()->GetInteractor();
        auto ren = inter->GetRenderWindow();
        ren->Finalize();
        inter->TerminateApp();
    }
    catch(...)
    {

    }

    //Accept the close event, closing the app
    event->accept();
}

//This fires when we pick File->Exit
void MainWindow::on_actionExit_triggered()
{
    //Call closeEvent
    this->close();
}

void MainWindow::ShowStatus(string message)
{
    this->statusBar()->showMessage(QString::fromStdString(message));
}

void MouseMoveCallbackFunction(vtkObject* caller, unsigned long eid, void* clientdata, void *calldata)
{
    vtkInteractorStyleImage* style = static_cast<vtkInteractorStyleImage*>(caller);

    auto eventPos = style->GetInteractor()->GetEventPosition();

    auto mainActor = MainWindow::instance->GetActor();
    auto picker = MainWindow::instance->GetPicker();
    auto renderer = MainWindow::instance->GetRenderer();
    auto cornerAnn = MainWindow::instance->GetCornerAnnotation();
    auto image = mainActor->GetInput();

    picker->Pick(eventPos[0], eventPos[1], 0, renderer);

    // There could be other props assigned to this picker, so
    // make sure we picked the image actor
    vtkAssemblyPath* path = picker->GetPath();
    bool validPick = false;
    if (path)
    {
      vtkCollectionSimpleIterator sit;
      path->InitTraversal(sit);
      vtkAssemblyNode *node;
      for (int i = 0; i < path->GetNumberOfItems() && !validPick; ++i)
      {
        node = path->GetNextNode(sit);
        if (mainActor == vtkImageActor::SafeDownCast(node->GetViewProp()))
        {
          validPick = true;
        }
      }
    }
    if (!validPick)
    {
      style->GetInteractor()->Render();
      // Pass the event further on
      style->OnMouseMove();
      return;
    }

    //Get world position of the pick
    double pos[3];
    picker->GetPickPosition(pos);

    //Convert world position to voxel index...
    double origin[3];
    double spacing[3];
    image->GetOrigin(origin);
    image->GetSpacing(spacing);
    pos[0] = (pos[0] - origin[0]) / spacing[0];
    pos[1] = (pos[1] - origin[1]) / spacing[1];
    pos[2] = (pos[2] - origin[2]) / spacing[2];

    //Round it down to ints...
    int image_coordinate[3];
    image_coordinate[0] = vtkMath::Round(pos[0]);
    image_coordinate[1] = vtkMath::Round(pos[1]);
    image_coordinate[2] = 0;

    //Get value string with crazy VTK template tricks
    string valueString;
    switch (image->GetScalarType())
    {
      vtkTemplateMacro((vtkValueMessageTemplate<VTK_TT>(image, image_coordinate, valueString)));
      default:
        return;
    }

    //Add text on the bottom left with location and value
    string fullMessage = "Location: (" + to_string(image_coordinate[0]) + ", " +
                                         to_string(image_coordinate[1]) + ", " +
                                         to_string(image_coordinate[2]) + ")\nValue: (" + valueString + ")";
    cornerAnn->SetText(0, fullMessage.c_str());

    //Get current windowing info, and add text on bottom right with WW and WC
    auto imageProperty = MainWindow::instance->GetActor()->GetProperty();
    if(imageProperty != nullptr)
    {
        auto window = imageProperty->GetColorWindow();
        auto level = imageProperty->GetColorLevel();

        string windowMessage = "WW: (" + to_string(window) + ")\nWC: (" + to_string(level) + ")";
        cornerAnn->SetText(1, windowMessage.c_str());
    }

    style->GetInteractor()->Render();
    style->OnMouseMove();
}

void MainWindow::on_actionOpenFile_triggered()
{
    //getOpenFileName displays a file dialog and returns the full file path of the selected file, or an empty string if the user canceled the dialog
    //The tr() function makes the dialog language proof (chinese characters, etc)
    QString fileName = QFileDialog::getOpenFileName(this, tr("Pick a DICOM file"), QString(), tr("All files (*.*);;DICOM FILES (*.dcm)"));

    if(!fileName.isEmpty())
    {
        string fileNameStd = fileName.toStdString();

        //Read all DICOM files in the specified directory
        VTK_NEW(vtkDICOMImageReader, dicomReader);
        dicomReader->SetFileName(fileNameStd.c_str());
        dicomReader->Update();
        auto dicomImage = dicomReader->GetOutput();        

        VTK_NEW(vtkImageCast, castFilter);
        castFilter->SetInputData(dicomImage);
        castFilter->SetOutputScalarTypeToFloat();
        castFilter->Update();

        m_mainActor = vtkSmartPointer<vtkImageActor>::New();
        m_mainActor->GetMapper()->SetInputData(castFilter->GetOutput());
        m_mainActor->InterpolateOff();

        VTK_NEW(vtkCallbackCommand, mouseMoveCallback);
        mouseMoveCallback->SetCallback(MouseMoveCallbackFunction);

        m_picker = vtkSmartPointer<vtkPropPicker>::New();
        m_picker->PickFromListOn();
        m_picker->AddPickList(m_mainActor); //Give the picker a prop to pick

        m_cornerAnn = vtkSmartPointer<vtkCornerAnnotation>::New();
        m_cornerAnn->SetLinearFontScaleFactor(2);
        m_cornerAnn->SetNonlinearFontScaleFactor(1);
        m_cornerAnn->SetMaximumFontSize(14);
        m_cornerAnn->GetTextProperty()->SetColor(1, 1, 1);
        m_cornerAnn->SetImageActor(m_mainActor);

        VTK_NEW(vtkInteractorStyleImage, style);
        style->AddObserver(vtkCommand::MouseMoveEvent, mouseMoveCallback);

        VTK_NEW(vtkRenderWindowInteractor, interactor);
        interactor->SetRenderWindow(m_renderer->GetRenderWindow());
        interactor->SetInteractorStyle(style);

        m_renderer->RemoveAllViewProps();
        m_renderer->AddActor(m_mainActor);
        m_renderer->AddViewProp(m_cornerAnn);
        m_renderer->GetActiveCamera()->SetParallelProjection(1);
        m_renderer->ResetCamera();        

        interactor->Initialize();
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
//        ifstream inputFile(fileName.toStdString().c_str());

//        //If the file isn't open, show an error box
//        if(!inputFile)
//        {
//            QMessageBox::critical(this, tr("Error"), tr("Could not open file"));
//            return;
//        }

//        //Get the file size
//        inputFile.seekg(0, ios::end);
//        int fileSize = inputFile.tellg();
//        inputFile.seekg(0, ios::beg);

//        //Read the file into data
//        vector<uint8_t> data(fileSize);
//        inputFile.read((char*) &data[0], fileSize);

//        //Extract the data dimensions from the header
//        memcpy(&m_rawLength, &(data[16]), 4*sizeof(uint8_t));
//        memcpy(&m_rawWidth, &(data[20]), 4*sizeof(uint8_t));
//        memcpy(&m_rawDepth, &(data[24]), 4*sizeof(uint8_t));

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

void MainWindow::on_actionReset_view_triggered()
{
    m_renderer->ResetCamera();
    m_renderer->GetActiveCamera()->SetViewUp(0,1,0);
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
    tracer->SetCaptureRadius(1000000); //Always close the polyline
    tracer->On();

    //Keep track of this polyline (easiest way of cleaning it later)
    m_polylines.push_back(tracer);

    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionClear_polylines_triggered()
{
    for(int i = 0; i < m_polylines.size(); i++)
    {
        vtkSmartPointer<vtkImageTracerWidget> widget = m_polylines[i];
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
    //main image. Its usually 1 unit ahead of it so we can see the polylines in front of the image
    VTK_NEW(vtkTransform, trans);
    trans->Translate(0, 0, -1);

    VTK_NEW(vtkTransformPolyDataFilter, transFilter);
    transFilter->SetInputData(appendFilter->GetOutput());
    transFilter->SetTransform(trans);
    transFilter->Update();

    VTK_NEW(vtkContourTriangulator, triFilter);
    triFilter->SetInputData(transFilter->GetOutput());
    triFilter->Update();

    VTK_NEW(vtkLinearExtrusionFilter, extrusion);
    extrusion->SetInputData(triFilter->GetOutput());
    extrusion->SetCapping(1);
    extrusion->SetScaleFactor(1);
    extrusion->SetExtrusionTypeToNormalExtrusion();
    extrusion->SetVector(0, 0, 1);
    extrusion->Update();

//    VTK_NEW(vtkPolyDataMapper, mapper);
//    mapper->SetInputData(extrusion->GetOutput());

//    VTK_NEW(vtkActor, polyActor);
//    polyActor->SetMapper(mapper);

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
    whiteImage->AllocateScalars(VTK_FLOAT,1);

    // fill the image with foreground voxels:
    float inval = -2.0f;
    float outval = 2.0f;
    vtkIdType count = whiteImage->GetNumberOfPoints();
    for (vtkIdType i = 0; i < count; ++i)
    {
        whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
    }

    // polygonal data --> image stencil:
    VTK_NEW(vtkPolyDataToImageStencil, pol2stenc);
    pol2stenc->SetInputData(extrusion->GetOutput());
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

    m_polyLineActor = vtkSmartPointer<vtkImageActor>::New();
    m_polyLineActor->SetInputData(imgstenc->GetOutput());
    m_polyLineActor->GetProperty()->SetInterpolationTypeToNearest();
    m_polyLineActor->GetMapper()->BackgroundOff();
    //m_polyLineActor->SetOpacity(0.5);

    //m_renderer->RemoveAllViewProps();
    //m_renderer->AddActor(m_polyLineActor);
    //m_renderer->AddActor(polyActor);
    //m_renderer->ResetCamera();

    this->ui->qvtkWidget->repaint();

    on_actionClear_polylines_triggered();
}

void MainWindow::on_actionTest_Sobel_filter_triggered()
{
    //Get currently displayed imageData
    auto dicomImageData = m_mainActor->GetInput();

    //Apply our CUDA kernel to it
    auto outputData = testSobelFilter(dicomImageData);

    //Display results
    m_mainActor->SetInputData(outputData);

    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionTest_Gaussian_filter_triggered()
{
    //Get currently displayed imageData
    auto dicomImageData = m_mainActor->GetInput();

    //Apply our CUDA kernel to it
    auto outputData = testGaussianFilter(dicomImageData);

    //Display results
    m_mainActor->SetInputData(outputData);

    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionTest_edge_indicator_triggered()
{
    //Get currently displayed imageData
    auto dicomImageData = m_mainActor->GetInput();

    //Apply our CUDA kernel to it
    auto outputData = testEdgeIndicator(dicomImageData);

    //Display results
    m_mainActor->SetInputData(outputData);

    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionNormalize_image_to_0_1_triggered()
{
    //Get currently displayed imageData
    auto dicomImageData = m_mainActor->GetInput();

    auto scalarRangeLower = dicomImageData->GetScalarRange()[0];
    auto scalarRangeUpper = dicomImageData->GetScalarRange()[1];

    //Shift the image data to the [0,1] range
    VTK_NEW(vtkImageShiftScale, shiftFilter);
    shiftFilter->SetInputData(dicomImageData);
    shiftFilter->SetShift(-1 * scalarRangeLower); //Shifts the range so the lower bound = 0
    shiftFilter->SetScale(1.0f / (scalarRangeUpper - scalarRangeLower)); //Scales to range so the upper bound = 1
    shiftFilter->Update();

    //Shift our mapper window width/center to perfectly fit the [0,1] range
    m_mainActor->SetInputData(shiftFilter->GetOutput());
    m_mainActor->GetProperty()->SetColorWindow(1.0f);
    m_mainActor->GetProperty()->SetColorLevel(0.5f);

    this->ui->qvtkWidget->repaint();
}

void MainWindow::on_actionTest_level_sets_triggered()
{
    auto dicomImageData = m_mainActor->GetInput();
    auto polyLineData = m_polyLineActor->GetInput();

    //Apply our CUDA kernel to it
    initLevelSets(dicomImageData, polyLineData);
}
