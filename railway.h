#ifndef railway_h
#define railway_h
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
typedef enum{
    M_SUCCESS = 0,
    CAM1_OPEN_ERR = 1,
    CAM2_OPEN_ERR = 2,
    IMG_READ_ERR = 3,
    LIGHT_ERR = 4,
    CTR_LOAD_ERR = 5,
    MATCH_ERR = 6,
    CAM1_RST_ERR = 7,
    CAM2_RST_ERR = 8,
    CAM1_RESULT_ERR = 9,
    CAM2_RESULT_ERR = 10,
}RM_ERROR_CODE;

int RailwayMeasure(short &Offset1,short &Offset2,int &ret1,int &ret2);
int ResetOri(int &ret1,int &ret2);
int VideoBlurDetectCam1(cv::Mat &srcimg);
int VideoBlurDetectCam2(cv::Mat &srcimg);
#endif /* railway_h */

