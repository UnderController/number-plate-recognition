#ifndef OCR_H
#define OCR_H
#include<opencv.hpp>
#include<plate.h>
using namespace cv;
class charaterSeg
{
public:
    Mat image; //data
    cv::Rect position; //location
    charaterSeg(Mat img, cv::Rect pos){
        image=img;
        this->position=pos;
    }
    charaterSeg();
};

class OCR
{
private:
    std::vector<charaterSeg> charaters;

    CvANN_MLP ann;
    bool trained;
    int numberclass=30; //class number
    char  strCharacters[30]={'0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
    int charatersize=15;   //image size  input of the class this can be change as different features
    const int charnorsize=20;    //preprocess size of the charater this is set as a constant

    vector<char> charstr;  //save number str
    vector<Rect> charpos;  //save number position
public:

    OCR(){
    }

    // initial function
    void initial()
    {
        trained=false;
        Mat data;
        Mat classes;
        //get train data
        this->dataread(data,classes);
        //train the data with ann
        this->train(data,classes,10);
    }

    Mat preprocesschar(Mat input){
        int width=input.cols;
        int height=input.rows;
        int max=std::max(width,height);
        //define the wapaffine matrix
        Mat trans=Mat::eye(2,3,CV_32F);
        trans.at<float>(0,2)=max/2-width/2; //remind here divided by 2
        trans.at<float>(1,2)=max/2-height/2;
        Mat warpimage(max,max,input.type());
        cv::warpAffine(input,warpimage,trans,warpimage.size(),INTER_LINEAR,BORDER_CONSTANT,cv::Scalar(0));
        Mat out;
        resize(warpimage,out,Size(charnorsize,charnorsize),0,0,INTER_CUBIC);
        return out;
    }


    // segment the image
    vector<charaterSeg> ChaSegment(Plate plate){
        vector<charaterSeg> out;
        charpos.clear();
        charstr.clear();
        Mat input=plate.image;
        if(input.channels()==3)
        {
            cv::cvtColor(input,input,CV_BGR2GRAY);
        }

        //input image is a gray image which is a plate
        //First use the threshold to obtain the binary image
        Mat image_threshold;
        cv::threshold(input,image_threshold,80,255,cv::THRESH_BINARY_INV);
        //imshow("temp",image_threshold);

        //Next find the contours, findcontours is aimed to find the whilt block
        Mat image_contours;
        image_threshold.copyTo(image_contours);
        std::vector< std::vector<cv::Point> >contours;
        cv::findContours(image_contours,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

        //approximate each contour by rect
        Mat result;
        image_threshold.copyTo(result);
//        Mat result1;
//        image_threshold.copyTo(result1);
//        cv::cvtColor(result1,result1,CV_GRAY2BGR);
        for(int i=0;i<contours.size();i++)
        {
            Rect rc=cv::boundingRect(contours[i]);
            //varify the rect size
            Mat auxRoi(result,rc);
            if(verifysize(auxRoi)){
                //store the auxRoi and position
                //preprocess the charater in the size 20*20
                auxRoi=preprocesschar(auxRoi);

                //extract the feature
                Mat features=feature(auxRoi,charatersize);

                //predict
                int result=0;
                result=this->classify(features);
               // std::cout<<result<<" ";

               // std::cout<<this->strCharacters[result];
                out.push_back(charaterSeg(auxRoi,rc));
                charstr.push_back(strCharacters[result]);
                charpos.push_back(rc);
                //cv::rectangle(result1,rc,cv::Scalar(0,0,255));
            }
        }
       // imshow("charseg",result1);
        return out;
    }


    /*
     * Three Cirteria 1:ratio 2 height 3:number of nonzero
     */
    bool verifysize(Mat roi){
        const float aspect=45.0f/77.0f;
        float ratio=(float)roi.cols/(float)roi.rows;
        float minaspect=0.2;
        float aspecterr=0.3;
        float maxaspect=aspect*(1+aspecterr);
        if(ratio<minaspect ||ratio>maxaspect){
            return false;
        }
        int maxheight=30;
        int minheight=15;
        if(roi.rows>maxheight ||roi.rows<minheight) return false;
        float areapercent=0.8;
        int maxarea=areapercent*roi.cols*roi.rows;
        int roi_area=cv::countNonZero(roi);
        if(roi_area>maxarea)
        {
            return false;
        }
        return true;
    }



     /* Exact the features
     *Features: Horizontal and veritical histograms
     * This is feature is a good features
     */
    Mat ProjectFeatures(Mat input, int type){
        int size;
        if(type)
        {
            size=input.rows;
        }
        else
        {
            size=input.cols;
        }
        Mat features(1,size,CV_32FC1);
        for(int i=0;i<size;i++){
            int nonzero;
            if(type)
            {
                nonzero=cv::countNonZero(input.row(i));
            }
            else
            {
                nonzero=cv::countNonZero(input.col(i));
            }
            features.at<float>(0,i)=nonzero;
        }

        //normailize the hist
        //cv::normalize(features,features,1,0,cv::NORM_L1);
        double min,max;
        cv::minMaxLoc(features,&min,&max);
        if(max>0){
            features.convertTo(features,-1,1.0f/max,0);
        }
        return features;

    }

    Mat feature(Mat input,int size){
        int Horizontal=0;
        int veritical=1;
        Mat horhist=ProjectFeatures(input,Horizontal);
        Mat verhist=ProjectFeatures(input,veritical);

//        //resize the input;
        Mat lowdata;
        cv::resize(input,lowdata,Size(size,size));

        int numberfeature=horhist.cols+verhist.cols+lowdata.cols*lowdata.rows;
        Mat out(1,numberfeature,CV_32FC1);
        out=cv::Scalar::all(0);
        int j=0;
        for(int i=0;i<horhist.cols;i++){
            out.at<float>(0,j)=horhist.at<float>(0,i);
            j++;
        }
        for(int i=0;i<verhist.cols;i++){
            out.at<float>(0,j)=verhist.at<float>(0,i);
            j++;
        }
        for(int y=0;y<lowdata.rows;y++){
            for(int x=0;x<lowdata.cols;x++){
                out.at<float>(0,j)=(float)lowdata.at<uchar>(y,x);
                j++;
            }
        }
        return out;

    }

    void dataread(Mat &data,Mat &classes){
        FileStorage fs;
        fs.open("OCR.xml",FileStorage::READ);
        if(fs.isOpened()){
            fs["TrainingDataF15"]>>data;
            fs["classes"]>>classes;
        }
        else
        {
            std::cout<<"open error";
        }
    }

    void  train(Mat traindata, Mat trainclass, int hlayer)
    {
        Mat layers(1,3,CV_32SC1);
        layers.at<int>(0,0)=traindata.cols;
        layers.at<int>(0,1)=hlayer;
        layers.at<int>(0,2)=numberclass;
        ann.create(layers,CvANN_MLP::SIGMOID_SYM,1,1);
        //change the class to the format
        Mat classes(trainclass.rows,numberclass,CV_32FC1);
        for(int i=0;i<trainclass.rows;i++)
        {
            for(int j=0;j<numberclass;j++){
                if(j==trainclass.at<int>(i,0))
                {
                    classes.at<float>(i,j)=1;
                }
                else
                     classes.at<float>(i,j)=0;
            }
        }

        Mat weight(1,traindata.rows,CV_32FC1,cv::Scalar::all(1));
        ann.train(traindata,classes,weight);
        trained=true;
    }

    int classify(Mat input)
    {
        Mat out(1,numberclass,CV_32FC1);
        ann.predict(input,out);
        double max;
        Point maxloc;
        cv::minMaxLoc(out,0,&max,0,&maxloc);
        return maxloc.x;
    }

    /*
     *
     * get the string in order
     *
     */
    std::string str(){
        std::string result="";
        vector<int> orderindex;
        vector<int> xposition;
        for(int i=0;i<charpos.size();i++)
        {
            xposition.push_back(charpos[i].x);
            orderindex.push_back(i);
        }

        for(int i=0;i<xposition.size();i++){
            for(int j=i;j<xposition.size();j++)
            {
                if(xposition[i]>xposition[j])
                {
                    int temp;
                    temp=xposition[i];
                    xposition[i]=xposition[j];
                    xposition[j]=temp;
                    temp=orderindex[i];
                    orderindex[i]=orderindex[j];
                    orderindex[j]=temp;
                }
            }
        }
        for(int i=0;i<charstr.size();i++){
            result+=charstr[orderindex[i]];
        }
        return result;
    }
    //show the charater segment
    void showSegment(Plate plate){
        charaters=this->ChaSegment(plate);
        Mat result;
        plate.image.copyTo(result);
        if(result.channels()==1)
        {
            cv::cvtColor(result,result,CV_GRAY2BGR);
        }
        for(int i=0;i<charaters.size();i++)
        {
            cv::rectangle(result,charaters[i].position,cv::Scalar(0,0,255));
        }
        if(charaters.size()!=0){
             std::cout<<"\n";
             std::cout<<str();
        }
        imshow("charaterSeg",result);
    }
};

#endif // OCR_H
