#-------------------------------------------------
#
# Project created by QtCreator 2015-04-08T16:38:31
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = ANPR
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    platdetection.cpp \
    plate.cpp \
    ocr.cpp
INCLUDEPATH += D:\Qt\opencv\include\opencv
INCLUDEPATH += D:\Qt\opencv\include\opencv2
INCLUDEPATH += D:\Qt\opencv\include

LIBS += D:\Qt\opencv\lib\libopencv_calib3d243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_contrib243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_core243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_features2d243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_flann243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_gpu243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_highgui243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_imgproc243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_legacy243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_ml243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_nonfree243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_objdetect243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_photo243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_stitching243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_ts243.a
LIBS += D:\Qt\opencv\lib\libopencv_video243.dll.a
LIBS += D:\Qt\opencv\lib\libopencv_videostab243.dll.a

HEADERS += \
    platdetection.h \
    plate.h \
    ocr.h
