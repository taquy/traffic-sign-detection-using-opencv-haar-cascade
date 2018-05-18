#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <traffic.h>
int main()
{
    VideoCapture cap(0);
    Traffic *d = new Traffic();
    d->isDebug = true;
    Mat frame;
    while (1) {
        cap >> frame;
        d->taquy(frame);
        waitKey(1);
    }

    return - 1;
}
