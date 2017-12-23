#include "mainwindow.h"
#include "ui_mainwindow.h"

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

void MainWindow::on_actionOpenFile_triggered()
{
    //getOpenFileName displays a file dialog and returns the full file path of the selected file, or an empty string if the user canceled the dialog
    //The tr() function makes the dialog language proof (chinese characters, etc)
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Thorlabs Image"), QString(), tr("Text Files (*.txt)"));

    if(!fileName.isEmpty())
    {
        //Create an input file stream
        std::ifstream inputFile(fileName.toStdString().c_str());

        //If the file isn't open, show an error box
        if(!inputFile)
        {
            QMessageBox::critical(this, tr("Error"), tr("Could not open file"));
            return;
        }

        //Get the file size
        inputFile.seekg(0, std::ios::end);
        int fileSize = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);

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
    }
}

void MainWindow::on_actionExit_triggered()
{
    qApp->exit();
}
