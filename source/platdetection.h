#ifndef PLATDETECTION_H
#define PLATDETECTION_H
#include<opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<time.h>
#include<QDebug>
#include<plate.h>
#include<sstream>
using namespace cv;

/*
 * This class is used to detect the region
 * First use x-Sobel to detect the regions changing much
 * Second use the open and close morphology to detect the region,combine the small regions
 * Next varify the regions size, remove the ratio and the size are out of the range
 * Use floodfill method, because the backgroud of the plate, so hit one seed at the background, Then floodfill will
 * Fill the mask all the Background, and varify the size again
 * Finally use the SVM to detect the plate, 2 class problem
 *
 *Some problem remain: we have to train the svm every time, so it is better to save the major parames, and support vector and KERNEL_SMOOTH
 * In XML file
 */

class PlatDetection
{
private:
    // Mat  image
    Mat input;
    // store the regions detected
    std::vector<Plate> regions;
    // svm classifier
    CvSVM svmclassifier;
public:

    void setInput(Mat image){
        input=image;
    }

    //function detect the plat
    void detectRegion(){
        // clear the regions
        regions.clear();
        Mat gray;
        Mat grayDetect;
        if(input.channels()==3){
            cv::cvtColor(input,gray,CV_BGR2GRAY);
        }

        //first blur the image and use the sobel
        Mat blurImage;
        cv::GaussianBlur(gray,blurImage,Size(7,7),0);
        cv::Sobel(blurImage,grayDetect,CV_8U,1,0,3,1,0);
        cv::threshold(grayDetect,grayDetect,0,255,CV_THRESH_OTSU|CV_THRESH_BINARY);
        //imshow("sobel",grayDetect);

        // use the morphgraphy to detect the region
        //the size of close morphology may need to tune
        Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        cv::morphologyEx(grayDetect,grayDetect,cv::MORPH_OPEN,kernel);
        kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(20,3));
        cv::morphologyEx(grayDetect,grayDetect,cv::MORPH_CLOSE,kernel);
        //imshow("morphology",grayDetect);


        //find the counter and remove bad region according to size
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(grayDetect,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        //vaild each contours
        std::vector<std::vector<cv::Point> >::iterator it=contours.begin();
        //store the rotaterect
        std::vector<cv::RotatedRect> RotateRects;
        while(it!=contours.end())
        {
               cv::RotatedRect rotate_rec=cv::minAreaRect(Mat(*it));
               //vaild the region according to the size
               if(!verifySize(rotate_rec)){
                   it=contours.erase(it);
               }
               else
               {
                   RotateRects.push_back(rotate_rec);
                   it++;
               }
        }


        Mat result;
        input.copyTo(result);
       // cv::drawContours(result,contours,-1,Scalar(0,0,255));
       // imshow("varify the size",result);

        //remove the origin according to the white plate
        // some argument need to tune for example updiff and lodiff,seed
        int seedNumber=30;
        for(int i=0;i<RotateRects.size();i++){
            //choose the seed around the center
            RotatedRect rc=RotateRects[i];
            int minsize=(rc.size.width<rc.size.height)?rc.size.width:rc.size.height;
            minsize=minsize/2;
            circle(result,rc.center,1,cv::Scalar(255,0,0));
            //define the mask that using in the floodfill
            Mat mask(input.rows+2,input.cols+2,CV_8UC1);
            mask=cv::Scalar::all(0);
            int updiff=30;
            int lodiff=30;
            int newvalue=255;
            int connectity=4;
            int flag=connectity+(newvalue<<8)+CV_FLOODFILL_MASK_ONLY+CV_FLOODFILL_FIXED_RANGE;
            Rect cmp;
            srand(time(NULL));
            for(int j=0;j<seedNumber;j++){
                Point seed;
                seed.x=rc.center.x+rand()%minsize-minsize/2;
                seed.y=rc.center.y+rand()%minsize-minsize/2;
                circle(result,seed,1,cv::Scalar(0,255,0));
                cv::floodFill(input,mask,seed,cv::Scalar(255,0,0),&cmp,cv::Scalar(lodiff,lodiff,lodiff),
                              cv::Scalar(updiff,updiff,updiff),flag);

            }

            //check the mask  size again
            std::vector<cv::Point> points;
            Mat_<uchar>::iterator maskit=mask.begin<uchar>();
            Mat_<uchar>::iterator maskend=mask.end<uchar>();
            while(maskit!=maskend){
                if(*maskit==255){
                    points.push_back(maskit.pos()); // get the iteration location
                }
                maskit++;
            }

           RotatedRect maskrect=cv::minAreaRect(Mat(points));
            if(this->verifySize(maskrect)){
                qDebug()<<"verify success";
                qDebug()<<maskrect.size.area()<<maskrect.size.width;
                //get the rect points and draw lines
                cv::Point2f rectPoints[4];
                maskrect.points(rectPoints);
                for(int j=0;j<4;j++){
                    cv::line(result,rectPoints[j],rectPoints[(j+1)%4],cv::Scalar(0,0,255));
                }

                //success and roated the rect, save
                float angle=maskrect.angle;
                float r=(float)maskrect.size.width/(float)maskrect.size.height;
                if(r<1){
                    angle=angle+90;
                }
                Mat rotateMat=cv::getRotationMatrix2D(maskrect.center,angle,1);
                Mat rotate_img;
                //apply the rotated to the input matrix
                cv::warpAffine(input,rotate_img,rotateMat,input.size(),CV_INTER_CUBIC);
                //get the subrect
                Mat image_crop;
                Size  crop_size=maskrect.size;
                if(r<1){
                    std::swap(crop_size.width,crop_size.height);
                }
                cv::getRectSubPix(rotate_img,crop_size,maskrect.center,image_crop);

                //regular the subpix and equlization
                Size size_format(144,33);  //width height
                Mat image_format(size_format,CV_8UC3);
                cv::resize(image_crop,image_format,size_format,0,0,CV_INTER_CUBIC);
                Mat gray_format;
                cv::cvtColor(image_format,gray_format,CV_BGR2GRAY);
                //equlization hist
                cv::GaussianBlur(gray_format,gray_format,Size(3,3),0);
                cv::equalizeHist(gray_format,gray_format);

//                //store the candidate images
//                std::stringstream ss;
//                ss << "haha" << "_" << i << ".jpg";
//                imwrite(ss.str(),gray_format);

                //store the region rect into the regions Plate(image,rect) rect refers position
                this->regions.push_back(Plate(gray_format,maskrect.boundingRect()));
          }
//          imshow("temp",mask);
//          cv::waitKey();
        }
        //show the result
        //imshow("candidates",result);
        
    }

    /*
     * varify the size of the rotate rect
     * Many arguments need to tune for example err, and minheight and maxheight
     */
    bool verifySize(RotatedRect rec){
        //get the rec width and height
        float height=rec.size.height;
        float width=rec.size.width;
        float r=width/height;
        // be careful that check the width is larger than the height
        if(r<1){
            r=1/r;
        }
        const float ratio=52.0/11.0;
        float err=0.4;
        if(r<ratio*(1-err) || r>ratio*(1+err)){
            return false;
        }
        //varify the size
        int minheight=15;
        int maxheight=220;
        int minsize=minheight*ratio*minheight;
        int maxsize=maxheight*ratio*maxheight;
        int size=height*width;
        if(size<minsize || size>maxsize){
            return false;
        }
        return true;
    }

    /*
     * train the svm
     * The function also remind to method of reading xml in opencv
     * many parames to tune
     */
    void SVMTrain(){
        //the candidate region is store in the regions as a plant of Plate
        cv::FileStorage fs;
        fs.open("SVM.xml",FileStorage::READ);
        Mat trainData;
        Mat trainClass;
        if(fs.isOpened()){
            //read the train data
            qDebug()<<"training";
            fs["TrainingData"]>>trainData;
            fs["classes"]>>trainClass;
           CvSVMParams params;
           params.svm_type=CvSVM::C_SVC;
           params.kernel_type=CvSVM::LINEAR;
           params.C=2;
           params.degree=0;
           params.gamma=1;
           params.coef0=0;
           params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,1000,0.01);
           svmclassifier.train(trainData,trainClass,Mat(),Mat(),params);
           int cor=0;
           for(int i=0;i<trainData.rows;i++){
               int temp=(int)svmclassifier.predict(Mat(trainData.row(i)));
               //qDebug()<<temp;
               if(temp==trainClass.at<uchar>(i,0)){
                   cor++;
               }
           }
           qDebug()<<"Accuracy of the training"<<float(cor)*1.0/trainData.rows;
        }

    }


    /*
     * use the svm classifier to classifier the plate
     */
    void PlateClassifier(){
        qDebug()<<"candidate region number is"<<regions.size()<<endl;
        std::vector<Plate>::iterator it=regions.begin();
        while(it!=regions.end())
        {
            //predict the candidate plate  
           //Mat imagetest=(Mat)(*it).image.reshape(1,1); //no change the data make sure that the mat is contious

            Mat_<uchar>::iterator image_it=(*it).image.begin<uchar>();
            Mat_<uchar>::iterator image_end=(*it).image.end<uchar>();
            int imagetest_size=(*it).image.rows*(*it).image.cols;
            Mat imagetest(1,imagetest_size,CV_8UC1);
            Mat_<uchar>::iterator image_test=imagetest.begin<uchar>();
            while(image_it!=image_end && image_test!=imagetest.end<uchar>()){
                *image_test=*image_it;
                image_test++;
                image_it++;
            }
            imagetest.convertTo(imagetest,CV_32FC1);  //convert to the 32
            int response=(int)svmclassifier.predict(imagetest);
            qDebug()<<response<<endl;
            if(response!=1){
                it=regions.erase(it);
            }
            else{
                it++;
            }
        }
    }


    /*
     *
     *show the final detection region
     *
     */
    void drawResult(){
        std::vector<Plate>::iterator it=regions.begin();
        std::vector<Plate>::iterator end=regions.end();
        Mat result;
        input.copyTo(result);
        qDebug()<<"final detection number after SVM:"<<regions.size();
        while(it!=end){
            cv::rectangle(result,(*it).position,cv::Scalar(0,0,255));
            it++;
        }
        imshow("result",result);
    }

    std::vector<Plate> run(Mat input){
        this->setInput(input);
        detectRegion();
        SVMTrain();
        PlateClassifier();
        drawResult();
        //store the candidate images
//        std::stringstream ss;
//        for(int i=0;i<regions.size();i++)
//        {
//            ss << "haha" << "_" << i << ".jpg";
//            imwrite(ss.str(),((Plate)regions[i]).image);
//        }
        std::vector<Plate> temp=regions;
        return temp;

    }


    PlatDetection();
};

#endif // PLATDETECTION_H
