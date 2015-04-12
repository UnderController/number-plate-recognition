#include <QCoreApplication>
#include<platdetection.h>
#include<ocr.h>
using namespace cv;
//#define DEBUG
//#undef DEBUG
int main()
{
    std::string  filename="test.jpg";
    Mat image=imread(filename);

    if(!image.empty())
    {
        namedWindow("test");
        imshow("test",image);
    }
//    PlatDetection plateDetection;
//    plateDetection.setInput(image);;
//    plateDetection.detectRegion();
//    plateDetection.SVMTrain();
//    plateDetection.PlateClassifier();
//    plateDetection.drawResult();

    PlatDetection plateDetection;
    OCR ocr;
    ocr.initial();
    std::vector<Plate> plates=plateDetection.run(image);
    if(plates.size()>0)
    {
        for(int i=0;i<plates.size();i++)
        {
            Plate plate=plates[i];
            ocr.showSegment(plate);
            cv::waitKey();
        }
    }
    else
    {
        std::cout<<"No detect any plate";
    }

//    #ifdef DEBUG
//        std::cout<<"debug";
//    #else
//       std::cout<<"release";
//    #endif

    cv::waitKey();
}
