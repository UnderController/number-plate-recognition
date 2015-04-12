#ifndef PLATE_H
#define PLATE_H
#include<opencv.hpp>
using namespace cv;
class Plate
{
public:
    Mat image;
    cv::Rect position;
    Plate(Mat im,Rect pos);
    Plate();
};

#endif // PLATE_H
