#include "mainwindow.h"
#include <QApplication>
#include <QDebug>
#include <QStyleFactory>
#include <QVTKWidget.h>

int main(int argc, char** argv)
{
    //SetStyle needs to run before QApplication is created in order to get the
    //True color palette
    //QApplication::setStyle("Cleanlooks");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

