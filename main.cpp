#include <QApplication>
#include <QDebug>
#include <QStyleFactory>
#include <QVTKWidget.h>

#include "mainwindow.h"

int main(int argc, char** argv)
{
    //SetStyle needs to run before QApplication is created in order to get the
    //True color palette
    //QApplication::setStyle("Cleanlooks");
    QApplication a(argc, argv);

    MainWindow::GetInstance().show();

    a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));

    a.exec();

    return 0;
}

