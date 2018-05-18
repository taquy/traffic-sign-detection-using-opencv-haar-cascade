
#include "traffic.h"

#define debug

String modeldir = "/home/taquy/Projects/cpp/HaarGpu/models/";
String sign_model = modeldir + "all.xml";
String stop_model = modeldir + "stop.xml";
String left_model = modeldir + "left.xml";
String right_model = modeldir + "right.xml";

//RNG rng(12345);

Traffic::Traffic() {
    this->signName = {"Right", "Left", "Stop"};
    this->classifiers = {
        cuda::CascadeClassifier::create(right_model),
        cuda::CascadeClassifier::create(left_model),
        cuda::CascadeClassifier::create(stop_model),
    };


    // Distance calculate
    this->w1 = (double) 170; // px
    this->d1 = (double) 20; // cm

    // Scoring approximate
    this->isDetect = true;
    this->maxFrame = 1;
    this->crrFrame = 0;
    this->score = new int[3] {0,0,0};

    // Limit frame detection
    this->fpslimit = 1; // detect object every 30 frames

    this->signId = -1;

}

int Traffic::difficulty[3] = {15, 30, 30};

//                              T     R     B     L
double Traffic::boundbox[4] = {00.0, 00.0, 00.0, 00.0};


int Traffic::taquy(Mat frame) {
    this->crrFrame++;

    if (this->crrFrame % this->fpslimit != 0)
        return this->signId;
    this->crrFrame = 0;

#ifdef debug
   cout << "Checkpoint" << endl;
#endif

    this->signId = this->getID(frame);
    return this->signId;

}

void Traffic::reset() {
    this->isDetect = true;
    this->maxFrame = 10;
    this->crrFrame = 0;
    delete[] this->score;
    this->score = new int[3] {0,0,0};
}

Mat Traffic::splitter(Mat img) {
    Rect box = this->cropbox(img);
    return img(box);
}

Rect Traffic::cropbox(Mat frame) {
    double w = frame.cols;
    double h = frame.rows;
    double x = w * this->boundbox[3];
    double y = h * this->boundbox[0];
    double w2 = w - w * (this->boundbox[1] + this->boundbox[3]);
    double h2 = h - h * (this->boundbox[0] + this->boundbox[2]);
    return Rect(x, y, w2, h2);
}

double Traffic::calcDistance (double w2) {
    return (this->w1 * this->d1) / w2;
}

vector<Rect> Traffic::detectGpu(cuda::GpuMat grayMat, int signId) {
    grayMat = grayMat.clone();

    Ptr<cuda::CascadeClassifier> gpuCascade = this->classifiers[signId];

    vector<Rect> boxes;

    cuda::GpuMat gpuFound;

    gpuCascade->setMinNeighbors(this->difficulty[signId]);
    gpuCascade->detectMultiScale(grayMat, gpuFound);
    gpuCascade->convert(gpuFound, boxes);

    return boxes;
}


int Traffic::getID(Mat frame) {
    Mat area = frame.clone();
    area = this->splitter(area);

    cuda::GpuMat gpuMat, grayMat;

    gpuMat.upload(area);

    cuda::cvtColor(gpuMat, grayMat, COLOR_BGR2GRAY);

    int n = this->signName.size();

    vector<Rect> boxes1 = this->detectGpu(grayMat, Traffic::STOP);
    #ifndef debug
    if (boxes1.size() > 0) return Traffic::STOP;
    #endif

    vector<Rect> boxes2 = this->detectGpu(grayMat, Traffic::LEFT);

    #ifndef debug
    if (boxes2.size() > 0) return Traffic::LEFT;
    #endif

    vector<Rect> boxes3 = this->detectGpu(grayMat, Traffic::RIGHT);
    #ifndef debug
    if (boxes3.size() > 0) return Traffic::RIGHT;
    #endif

    vector<vector<Rect>> boxtrunk;
    boxtrunk.push_back(boxes1);
    boxtrunk.push_back(boxes2);
    boxtrunk.push_back(boxes3);

    if (this->isDebug) {
        frame = this->draw(frame, boxtrunk[0], this->signs[Traffic::STOP]);
        frame = this->draw(frame, boxtrunk[1], this->signs[Traffic::LEFT]);
        frame = this->draw(frame, boxtrunk[2], this->signs[Traffic::RIGHT]);
        this->drawRuler(frame);
        vector<vector<double>> distancesMult = this->calcDistanceMult(boxtrunk, frame);

        imshow("test", frame);


        if (boxes1.size() > 0) return Traffic::STOP;
        if (boxes2.size() > 0) return Traffic::LEFT;
        if (boxes3.size() > 0) return Traffic::RIGHT;
    }

    return -1;
}

double Traffic::calcAngle(Mat frame, Rect box, Point s) {
    double w = frame.cols;
    double h = frame.rows;
    Point O = Point(w/2, h/2);
    Point M = Point(w/2, h);
    Point X = Point(box.width/2 + s.x, box.height/2 + s.y);
    // calculate vectors
    Point OM = Point(M.x - O.x, M.y - O.y);
    Point XM = Point(M.x - X.x, O.y - X.y);
    // calculate dot
    double uv = OM.x * XM.x + OM.y * XM.y;
    double OMd = sqrt(pow(OM.x, 2) + pow(OM.y, 2));
    double XMd = sqrt(pow(XM.x, 2) + pow(XM.y, 2));
    double uvd = OMd * XMd;
    return acos(uv / uvd) * (180 / M_PI);
}


vector<vector<double>> Traffic::calcDistanceMult(vector<vector<Rect>> boxtrunk, Mat frame) {
    vector<vector<double>> distancesMult;
    for (int i = 0; i < boxtrunk.size(); i++) {
        vector<Rect> boxes = boxtrunk[i];
        vector<double> distances;
        for (int j = 0; j < boxes.size(); j++) {
            Rect box = boxes[j];
            double w = this->calcDistance(box.width);
            w = round(w);
            distances.push_back(w);
            if (this->isDebug) {
                Point a(box.x, box.y + box.height + 15);
                string label = "Distance: " + to_string((int) w);
                Scalar clrf = Scalar(0,0,255);
                putText(frame, label, a, FONT_HERSHEY_SIMPLEX, 1, clrf, 2, 0, false);
            }
        }
        distancesMult.push_back(distances);
    }
    return distancesMult;
}


Mat Traffic::draw(Mat frame, vector<Rect> boxes, String label) {
    // draw frontier line
    Rect box = this->cropbox(frame);
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255,255,255), 2);


    // draw rects
    for( size_t i = 0; i < boxes.size(); i++ )
    {
        int x = boxes[i].x + box.x;
        int y = boxes[i].y + box.y;
        Point a(x, y);
        Point b(x + boxes[i].width, y + boxes[i].height);

        rectangle(frame, a, b, Scalar(0, 255, 0), 3);
        this->drawAngle(frame, boxes[i], a);

        Scalar clrf = Scalar(255,0,0);
        putText(frame, label, a, FONT_HERSHEY_SIMPLEX, 1, clrf, 3, 0, false);
    }
    return frame;
}

void Traffic::drawAngle(Mat frame, Rect box, Point s) {
    Point hzs = Point(0 + box.x, box.width / 2 + box.y);
    Point hze = Point(box.width + s.x, box.height / 2 + s.y);

    Point vts = Point(box.width / 2  + box.x, 0 + box.y);
    Point vte = Point(box.width / 2 + s.x, box.height + s.y);

    Scalar clr = Scalar(0,255,0);
    cv::line(frame, hzs, hze, clr, 2, 8, false);
    cv::line(frame, vts, vte, clr, 2, 8, false);

    // draw centroit
    Point ctp = Point(box.width / 2 + s.x, box.height / 2 + s.y);
    circle(frame, ctp, 5, clr, 2, 8, false);

    // draw connect line
    clr = Scalar(0,0,255);
    Point ctpf = Point(frame.cols / 2, frame.rows / 2);
    cv::line(frame, ctp, ctpf, clr, 2, 8, false);
    // Draw text
    string label = "Angle: " + to_string(calcAngle(frame, box, s));
    Scalar clrf = Scalar(255,0,255);
    Point drt = Point(box.x, box.y + box.height + 45);
    putText(frame, label, drt, FONT_HERSHEY_SIMPLEX, 1, clrf, 2, 0, false);
}

void Traffic::drawRuler(Mat frame) {
    // draw ruler
    Point hzs = Point(0, frame.rows / 2);
    Point hze = Point(frame.cols, frame.rows / 2);

    Point vts = Point(frame.cols / 2, 0);
    Point vte = Point(frame.cols / 2, frame.rows);

    Scalar clr = Scalar(222,222,115);
    cv::line(frame, hzs, hze, clr, 2, 8, false);
    cv::line(frame, vts, vte, clr, 2, 8, false);

    // draw centroit
    Point ctp = Point(frame.cols / 2, frame.rows / 2);
    circle(frame, ctp, 5, clr, 2, 8, false);
}
