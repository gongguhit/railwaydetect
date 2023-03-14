#ifndef RAIL_CALC_H
#define RAIL_CALC_H

#include<iostream>
#include"opencv2/opencv.hpp"
#include"opencv2/core.hpp"

//2020.4.16 添加白平衡
#include<opencv2/ml/ml.hpp>
using namespace cv::ml;
using namespace std;
using namespace cv;
//Mac测试路径
//static string init_address = "/home/gonggu/0.00.jpg";
//static string init_address2 = "/home/gonggu/4.33.jpg";
//开发板上用路径
static string init_address = "/home/root/camera1_init.jpg";
static string init_address2 = "/home/root/camera2_init.jpg";
static int roi = 1000;

//offline批量测试pattern
//static string pattern_img = "/Users/gonggu/Documents/railphoto/*.bmp";

//用到的结构体
struct zhixin{
    Point2f position;
    unsigned int width;
    int index;
    bool isupside = false;
    double prop = 0;
    Point3f match_position=Point3f(-1,-1,-1);
    Mat T;
};
struct value_index
{
    float value;
    int index;
};
struct ruler
{
    int position_index;
    Point3f position;
    bool isupside=false;
    double prop = 0;
};


int preprocessing(Mat &img_src,Mat &MDBlur);
int Contours_right_size(Mat img, vector<vector<Point>> &contours_right_size);
int Contours_is_rect(vector<vector<Point>> &contours_right_size,vector<vector<Point>> &contours_is_rect);
vector<Point2f>calc_moment(vector<vector<Point>> contours_is_rect);
vector<unsigned int>cal_contours_width(vector<vector<Point>> contours_is_rect);
double cal_mid_line(vector<vector<Point>> contours_is_rect);
bool cmp(struct value_index a,struct value_index b);
int find_absmin(vector<double> &a, double &k);
vector<int> match_position(vector<struct zhixin> zx, vector<struct ruler> bc);
int offsetlab(vector<int> match_result, vector<struct zhixin> &zx, vector<struct ruler> &bc, vector<int> &offset_lab);
Mat img_load(string address);
int getphoto(Mat &frame);
int getphoto2(Mat &frame);
int camera_calibrate();
int camera2_calibrate();
int calc_T(vector<struct zhixin> &zx1, vector<struct zhixin> &zx2);
double calcdistance(vector<struct zhixin> &zx1,vector<struct zhixin> &zx2);
int image_processing2(Mat &img,vector<struct zhixin> &zx_final);
int getzx_camera1(vector<struct zhixin> &zx);
int getzx_camera2(vector<struct zhixin> &zx);
int resultol_camera1(short &result);
int resultol_camera2(short &result);
Mat PerfectReflectionAlgorithm(Mat src);
float getrulerslope(vector<struct zhixin> zx);
float avgslope(vector<Point2f> ruler1, vector<Point2f> ruler2);
#endif

