#ifndef TRAFFIC_H
#define TRAFFIC_H


#include <iostream>
#include <math.h>

using namespace std;

#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>

using namespace cv;

class Traffic
{

    private:
        vector<Ptr<cuda::CascadeClassifier>> classifiers;

        // Distance calculate
        double w1;
        double d1;

        // Image process
        Mat splitter(Mat img);
        Rect cropbox(Mat img);

        // Scoring approximate

        int maxFrame;
        int crrFrame;
        int *score;

        // Limit frame detection
        int fpslimit;

    public:
      Traffic();
      vector<String> signName;


      int getID(Mat frame);
      int taquy(Mat frame);
      void reset();
      vector<Rect> detectGpu(cuda::GpuMat grayMat, int signId);
      Mat draw(Mat frame, vector<Rect> boxes, String label);

      double calcDistance (double w2);
      vector<vector<double>> calcDistanceMult(vector<vector<Rect>> boxtrunk, Mat frame);

      double calcAngle(Mat frame, Rect box, Point s);
      void drawRuler(Mat frame);
      void drawAngle(Mat frame, Rect box, Point s);

      static const int  RIGHT = 0;
      static const int LEFT  = 1;
      static const int STOP = 2;

      static double boundbox[4];
      static int difficulty[3];

      bool isDetect;
      bool isDebug;

      int signId;

      string signs[5] = {
          "Right",              // 0
          "Left",              // 1
          "Stop",             // 2
          "No sign detected" // 3
      };
};


#endif // TRAFFIC_H
