#include"rail_calc.h"
#include"opencv2/highgui.hpp"
#include<vector>
#include<algorithm>
#include"railway.h"
#include<unistd.h>
#include<time.h>
#include<string>

double camera[9] = {17920,0,640,0,17920,540,0,0,1};
double dist[5] = {-0.4501,20.8062,-0.0496,0.0343,-201.2331};
static Mat camera_matrix = Mat(3,3,CV_64FC1,camera);//相机内参矩阵
static Mat distCoeffs = Mat(5,1,CV_64FC1,dist);//相机畸变系数矩阵
using namespace std;

//图像预处理函数
int preprocessing(Mat &img_src,Mat &MDBlur)
{
    if(!img_src.data)
    {
        return IMG_READ_ERR;
    }
    img_src = PerfectReflectionAlgorithm(img_src);
    Mat graytu1;
    cvtColor(img_src, graytu1, COLOR_BGR2GRAY);
    Mat bil;
    bilateralFilter(graytu1, bil, 20, 10, 20);
    Mat adp;
    adaptiveThreshold(bil, adp, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 317, -7);
    MDBlur = adp;
    return M_SUCCESS;
}
//轮廓提取与筛选函数
int Contours_right_size(Mat img, vector<vector<Point>> &contours_right_size)
{
    //2019.11.15 代码优化，调整大小筛选方法
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++)
    {
        int tmp = hierarchy[i][3];
        if(tmp==-1&&(contours[i].size()>300)&&contours[i].size()<2000)
        {
            contours_right_size.push_back(contours[i]);
        }
    }
    if(contours_right_size.size()==0)
    {
        return 4;//min or max contours size not suitable
    }
    else return M_SUCCESS;
}
int Contours_is_rect(vector<vector<Point>> &contours_right_size,vector<vector<Point>> &contours_is_rect)
{
    //2019.11.15优化
    vector<vector<Point2f>> min_area_rect;
    for(unsigned int i=0;i<contours_right_size.size();i++)
    {
        RotatedRect box;
        box = minAreaRect(contours_right_size[i]);
        Point2f vtx[4];
        box.points(vtx);
        vector<Point2f> vtx_vector;
        for(int j=0;j<4;j++)
        {
            vtx_vector.push_back(vtx[j]);
        }
        min_area_rect.push_back(vtx_vector);
    }
    double prop_list[contours_right_size.size()];
    for(int i =0 ;i<contours_right_size.size();i++)
    {
        prop_list[i] = matchShapes(min_area_rect[i], contours_right_size[i], 1, 0.0);
        if(prop_list[i]<0.04)
        {
            contours_is_rect.push_back(contours_right_size[i]);
        }
    }
    if(contours_is_rect.size()==0)
    {
        return 5;//no rect contours, check ad
    }
    else {
        return M_SUCCESS;
    }
}
//计算轮廓的质心
vector<Point2f>calc_moment(vector<vector<Point>> contours_is_rect)
{
    vector<Moments>mu(contours_is_rect.size());
    //计算轮廓的矩
    for(unsigned int i=0;i<contours_is_rect.size();i++)
    {
        mu[i] = moments(contours_is_rect[i],false);
    }
    //计算各个轮廓的质心
    vector<Point2f>mc(contours_is_rect.size());
    for(unsigned int i=0;i<contours_is_rect.size();i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
    return mc;
}
//计算轮廓的x方向边长，即为其对应格子的宽度
vector<unsigned int>cal_contours_width(vector<vector<Point>> contours_is_rect)
{
    //选用轮廓的垂直方向外接矩形来计算边长
    vector<Rect> boundRect(contours_is_rect.size());
    for(unsigned int i=0;i<contours_is_rect.size();i++)
    {
        boundRect[i] = boundingRect(contours_is_rect[i]);
    }
    //建立向量保存各边的边长
    vector<unsigned int>bound_width(contours_is_rect.size());
    for(unsigned int i=0;i<contours_is_rect.size();i++)
        {
            bound_width[i] = boundRect[i].width;
        }
    return bound_width;
}
//计算上下组轮廓的大致中心线位置，用于区分是上轮廓还是下轮廓
double cal_mid_line(vector<vector<Point>> contours_is_rect)
{
    vector<Rect> boundRect(contours_is_rect.size());
    for(unsigned int i=0;i<contours_is_rect.size();i++)
    {
        boundRect[i] = boundingRect(contours_is_rect[i]);
    }
    //建立向量保存各个矩形的左上角顶点y值
    //经过计算后的中心线在上边一组矩形的中点高度
    vector<unsigned int>bound_y(contours_is_rect.size());
    vector<unsigned int>bound_height(contours_is_rect.size());
    for(unsigned int i=0;i<contours_is_rect.size();i++)
        {
            bound_y[i] = boundRect[i].y;
            bound_height[i] = boundRect[i].height;
        }
    double tmp_sum=0;
    for(unsigned int i=0;i<contours_is_rect.size();i++)
    {
        tmp_sum +=bound_height[i];
    }
    double avg_height;
    avg_height = tmp_sum/contours_is_rect.size();
    int max_height = *max_element(bound_y.begin(),bound_y.end());
    int min_height = *min_element(bound_y.begin(),bound_y.end());
    double bound_mid_pos = (min_height+avg_height+max_height)/2;
    return bound_mid_pos;
}

//排序函数，用于各种排序
//cmp函数，用于输入进sort()中，a<b为正序排序
bool cmp(struct value_index a,struct value_index b)
{
    if(a.value<b.value)
    {
        return true;
    }
    return false;
};
//排序函数
template<typename T>
T sort_indexes(vector<size_t> &idx,vector<T> &v)
{
    value_index* a=new value_index[v.size()];
    for(int i=0;i<v.size();i++)
    {
        a[i].value=v[i];
        a[i].index=i;
    }
    sort(a,a+v.size(),cmp);
    for(int i=0;i<v.size();i++)
    {
        idx.push_back(a[i].index);
    }
    delete[] a;
    return 0;
}

//质心位置与标尺对应位置匹配函数
//首先给出一个寻找最小差值绝对值函数
int find_absmin(vector<double> &a, double &k)
{
    vector<double> abs_delta(a.size());
    for(int i=0;i<a.size();i++)
    {
        abs_delta[i] = abs(a[i]-k);
    }
    vector<size_t> idx;
    sort_indexes(idx, abs_delta);
    return idx[0];
}
//输入计算后的质心向量zx和标尺向量bc，返回与标尺相匹配的index
vector<int> match_position(vector<struct zhixin> zx, vector<struct ruler> bc)
{
    int zx_length = zx.size();
    int bc_length = bc.size();
    vector<int> match_result(zx_length);
    //现将标尺的上下比例信息提取到向量中
    vector<double> bc_prop(bc_length);
    for(int i=0;i<bc_length;i++)
    {
        bc_prop[i] = bc[i].prop;
    }
    for(int i=0;i<zx_length;i++)
    {
        int index_tmp;
        if(zx[i].prop!=0)//表示上标签
        {
            index_tmp= find_absmin(bc_prop,zx[i].prop);
        }
        else
        {
            index_tmp = -1;
        }
        match_result[i] = index_tmp;
    }
    return match_result;
}
//根据测试结果，会出现有多个点匹配到同一个benchmark的情形
//进行一步筛选，准则为保留与benchmark最接近的实际数值

int offsetlab(vector<int> match_result, vector<struct zhixin> &zx, vector<struct ruler> &bc, vector<int> &offset_lab)
{
    int match_num = match_result.size();
    for(int i=0;i<match_num;i++)
    {
        if(match_result[i]!=-1)
        {
            vector<int> tmp_label;//匹配相同的标尺位置的实际位置标签
            for(int j=0;j<match_num;j++)
            {
                if(match_result[j]==match_result[i])
                {
                    tmp_label.push_back(j);
                }
            }
            if(tmp_label.size()>=2)
            {
                vector<double> absdiff(tmp_label.size());
                for(int k=0;k<tmp_label.size();k++)
                {
                    absdiff[k]=abs(zx[tmp_label[k]].prop-bc[match_result[i]].prop);
                }
                vector<size_t> idx_temp;
                sort_indexes(idx_temp, absdiff);
                for(int n=1;n<absdiff.size();n++)
                {
                    offset_lab.push_back(tmp_label[idx_temp[n]]);
                }
            }
        }
    }
    //由于offset_lab中含有大量相同标签数，下面相同元素只保留一项
    sort(offset_lab.begin(),offset_lab.end());
    offset_lab.erase(unique(offset_lab.begin(),offset_lab.end()),offset_lab.end());
    if(offset_lab.size()==match_num)
    {
        return 6;
    }
    else {
        return M_SUCCESS;
    }
}
//solvepnp空间解算，获得所求质心与对应空间基准点的旋转和平移矩阵
//solvepnp所需要参数：目标点的空间坐标； 目标物体在图像上的坐标；相机内参矩阵；相机畸变系数
//2019.11.15 优化最终位移解算函数
//计算分别以此质心为锚点的T矩阵
int calc_T(vector<struct zhixin> &zx1, vector<struct zhixin> &zx2)
{
    int errorcode = M_SUCCESS;
    zx2 = zx1;
    vector<Point2f>zxtmp;
    for(int i=0;i<zx1.size();i++)
    {
        if(zx1[i].match_position!=Point3f(-1,-1,-1))
        {
            zxtmp.push_back(zx1[i].position);
        }
    }
    for(int i=0;i<zx1.size();i++)
    {
        if(zx1[i].match_position!=Point3f(-1,-1,-1))
          {
              Mat rtmp,ttmp;
              //建立向量，用于暂存作为输入pnp的三维点组
              vector<Point3f> pointtmp;
        for(int j=0;j<zx1.size();j++)
          {
              if(zx1[j].match_position!=Point3f(-1,-1,-1))
              {
                  float x;
                  float y;
                  x = zx1[j].match_position.x-zx1[i].match_position.x;
                  y = zx1[j].match_position.y-zx1[i].match_position.y;
                  pointtmp.push_back(Point3f(x,y,0));
              }
          }
              solvePnP(pointtmp, zxtmp, camera_matrix, distCoeffs, rtmp, ttmp);
              zx2[i].T = ttmp;
        }
    }
    return errorcode;
}
// 解算函数
double calcdistance(vector<struct zhixin> &zx1,vector<struct zhixin> &zx2)
{
    //根据求解算法公式，首先求解右侧比例系数
    //首先确定两幅图片用于匹配的对应点
    //建立pair结构体，用于保存相同位置的质心位置索引
    struct match_pair_index{
        int first;
        int second;
    };
    vector<struct match_pair_index> match_pair;
    for(int i=0;i<zx1.size();i++)
    {
        if(zx1[i].match_position!=Point3f(-1,-1,-1))
        {
            for(int j=0;j<zx2.size();j++)
            {
                if(zx2[j].match_position!=Point3f(-1,-1,-1))
                {
                    if(zx1[i].match_position==zx2[j].match_position)
                    {
                        struct match_pair_index matchtmp;
                        matchtmp.first = i;
                        matchtmp.second = j;
                        match_pair.push_back(matchtmp);
                    }
                }
            }
        }
    }
    //建立两个向量用于保存两组相关联的T矩阵
    vector<Mat> T1;
    vector<Mat> T2;
    for(int i=0;i<match_pair.size();i++)
    {
        T1.push_back(zx1[match_pair[i].first].T);
        T2.push_back(zx2[match_pair[i].second].T);
    }
    int Tnum = match_pair.size();
    double sum=0;
    int sign = 1;
    bool flag = true;
    for(int i=0;i<Tnum;i++)
    {
        for(int j=i+1;j<Tnum;j++)
        {
            if(zx2[match_pair[0].second].position.x-zx1[match_pair[0].first].position.x<0&&flag==true)
            {
                sign = -1;
                flag =false;
            }
            double lij =sqrt(pow(zx1[match_pair[i].first].match_position.x-zx1[match_pair[j].first].match_position.x,2)+pow(zx1[match_pair[i].first].match_position.y-zx1[match_pair[j].first].match_position.y, 2));
            Mat den_vec1 = zx1[match_pair[i].first].T-zx1[match_pair[j].first].T;
            double dis1 = sqrt(pow(den_vec1.at<double>(0,0),2)+pow(den_vec1.at<double>(1,0),2)+pow(den_vec1.at<double>(2,0),2));
            Mat den_vec2 = zx2[match_pair[i].second].T-zx2[match_pair[j].second].T;
            double dis2 = sqrt(pow(den_vec1.at<double>(0,0),2)+pow(den_vec1.at<double>(1,0),2)+pow(den_vec1.at<double>(2,0),2));
            sum+=lij/(dis1+dis2);
        }
    }
    //2020.4.22 计算标尺倾斜角度，得出沿前进方向位移
    //首先保存两组对应的质心
    vector<Point2f> zzx1;
    vector<Point2f> zzx2;
    for(int i=0;i<match_pair.size();i++)
    {
        zzx1.push_back(zx1[match_pair[i].first].position);
        zzx2.push_back(zx2[match_pair[i].second].position);
    }
    // 接下来分别求取两张图片中标尺的倾斜角度
    float ruler_slope1=getrulerslope(zx1);
    float ruler_slope2=getrulerslope(zx2);
    //（后续优化）若两个倾斜角度相差过大，说明标尺发生了旋转，报错
    // 选取平均值为标尺倾斜角度计算值
    float ruler_slope = (ruler_slope1+ruler_slope2)/2;
    // 获取标尺移动后与原标尺对应的质心连线斜率
    float cmp_slope = avgslope(zzx1, zzx2);
    // 斜率差值即为在标尺方向上的运动方向与水平面斜率
    float diff_slope = cmp_slope - ruler_slope;
    //将斜率转换为弧度值
    float slope_arctan = atan(diff_slope);
    // 位移缩放比值即位cos弧度值
    float result_prop = cos(slope_arctan);
    //计算lambda
    double prop=4*sum/(Tnum*(Tnum-1));
    //接下来计算测量位移值xi
    double xisum=0;
    for(int i=0;i<Tnum;i++)
    {
        Mat den_vec1 = zx1[match_pair[i].first].T-zx2[match_pair[i].second].T;
        double dis;
        dis = sqrt(pow(den_vec1.at<double>(0,0),2)+pow(den_vec1.at<double>(1,0),2)+pow(den_vec1.at<double>(2,0),2));
        xisum+=dis;
    }
    //计算最终结果
    double result;
    result = xisum*prop/Tnum;
    result = result*sign*result_prop;
    return result;
}
// 整合函数,输入为图片Mat

int image_processing2(Mat &img,vector<struct zhixin> &zx_final)
{
    int errorcode;
    Mat gray;
    errorcode = preprocessing(img,gray);
    if(errorcode!=M_SUCCESS)
    {
        
        return errorcode;
    }
    //图像轮廓提取
    vector<vector<Point>> contours_right_size;
    errorcode = Contours_right_size(gray,contours_right_size);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    vector<vector<Point>> contours_is_rect;
    errorcode = Contours_is_rect(contours_right_size,contours_is_rect);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    //求解轮廓组的质心，对应刻度的宽度
    int contours_num = contours_is_rect.size();
    vector<Point2f> zhixin;
    vector<unsigned int> bounding_width;
    zhixin = calc_moment(contours_is_rect);
    bounding_width = cal_contours_width(contours_is_rect);
    //求解刻度中心分割线位置，用于区分上下刻度
    int mid_line;
    mid_line = cal_mid_line(contours_is_rect);
    //将所得到的数据写入质心结构体中
    //所构成的zx是一个包含着所有质心结构体的向量
    vector<struct zhixin> zx(contours_num);
    for(int i=0;i<contours_num;i++)
    {
        zx[i].position = zhixin[i];
        zx[i].width = bounding_width[i];
        //*test*
        //cout<<zx[i].position<<endl;
    }
    //保存所有质心的x方向坐标
    vector<float> zhixin_x(contours_num);
    for(int i=0;i<contours_num;i++)
    {
        zhixin_x[i] = zhixin[i].x;
    }
    //由于最左和最右侧的两个轮廓有可能是非完全的，于是需要进行识别并去除
    //为了能够实现去除，首先需要对刻度的左右位置进行排序
    vector<size_t>idx;
    sort_indexes(idx,zhixin_x);//返回的标签形式为idx[k]的值是从小到大第k大所对应的标签值
    int contours_num2 = contours_num-2;
    //将刻度的x方向坐标大小标签记入结构体中
    for(int i=0;i<contours_num;i++)
    {
        int tmpk;
        tmpk = idx[i];
        zx[tmpk].index = i;
    }
    //将原质心结构体进行重新排序，并放入新的质心结构向量中
    vector<struct zhixin> zx2(contours_num2);
    for(int i=0;i<contours_num2;i++)
    {
        for(int j=0;j<contours_num;j++)
        {
            if(zx[j].index==i+1&&zx[j].index!=0&&zx[j].index!=contours_num) //去除了最左和最右侧两个轮廓
            {
                zx2[i].position = zx[j].position;
                zx2[i].width = zx[j].width;
            }
        }
        if(zx2[i].position.y<mid_line)
            zx2[i].isupside = true;
        zx2[i].index = i;
    }
    //此时的zx2便是从左至右并已排好序的质心结构体
    //接下来对所有的上-下组宽度计算比值
    for(int i=0;i<contours_num2;i++)
    {
        if(zx2[i].isupside==true && i!=contours_num2-1)
        {
            zx2[i].prop = (double)zx2[i].width/zx2[i+1].width;
        }
        //*test*
        //cout<<zx2[i].prop<<endl;
    }
    //对比所设计的标尺，比较上一步计算的上-下组宽度比值结果，得到对应的匹配位置
    //下面首先根据所选用的标尺形式进行构造
    //标尺结构体向量
    vector<struct ruler> benchmark(14);
    //标尺刻度编号index
    for(int i=0;i<14;i++)
    {
        benchmark[i].position_index = i;
    }
    //定义上下刻度
    for(int i=1;i<14;i=i+2)
    {
        benchmark[i].isupside = true;
    }
    //输入标尺刻度坐标
    //11.27新标尺
    benchmark[0].position = Point3f(4,22.5,0);
    benchmark[1].position = Point3f(21,7.5,0);
    benchmark[2].position = Point3f(39,22.5,0);
    benchmark[3].position = Point3f(56,7.5,0);
    benchmark[4].position = Point3f(74,22.5,0);
    benchmark[5].position = Point3f(91,7.5,0);
    benchmark[6].position = Point3f(109,22.5,0);
    benchmark[7].position = Point3f(126,7.5,0);
    benchmark[8].position = Point3f(144,22.5,0);
    benchmark[9].position = Point3f(160,7.5,0);
    benchmark[10].position = Point3f(179,22.5,0);
    benchmark[11].position = Point3f(196,7.5,0);
    benchmark[12].position = Point3f(214,22.5,0);
    benchmark[13].position = Point3f(231,7.5,0);
    //输入对应所有上侧刻度的比例信息
    benchmark[0].prop = 0.4;
    benchmark[2].prop = (double)5/9;
    benchmark[4].prop = 0.75;
    benchmark[6].prop = 1;
    benchmark[8].prop = (double)16/12;
    benchmark[10].prop = (double)1.8;
    benchmark[12].prop = 2.5;
    //标尺定义过后，进行位置匹配
    vector<int> match_result(contours_num2);
    match_result = match_position(zx2,benchmark);
    //重合点筛选：有可能会有多组zx2对象[i1][i2]...匹配到同一个标尺位置benchmark[k]
    vector<int>offset_lab;
    errorcode = offsetlab(match_result,zx2,benchmark,offset_lab);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    //去除过匹配点
    for(int i=0;i<offset_lab.size();i++)
    {
        match_result[offset_lab[i]] = -1;
    }
    //2019.11.01 若匹配点只有一组，报错
    int nomatchedcount=0;
    for(int i=0;i<match_result.size();i++)
    {
        if(match_result[i]==-1)
        {
            nomatchedcount+=1;
        }
    }
    if(nomatchedcount>=match_result.size()-1)
    {
        errorcode = MATCH_ERR;
        return errorcode;
    }
    //将匹配结果录入质心结构体中
    //2019.11.10 修改只有通过match标号不等于-1的才用于pnp
    for(int i =0;i<contours_num2-1;i++)
    {
        if(zx2[i].isupside==true)
        {

            zx2[i].match_position=benchmark[match_result[i]].position;
            zx2[i+1].match_position = benchmark[match_result[i]+1].position;
        }
    }
    errorcode = calc_T(zx2, zx_final);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    return 0;
}
// img loading function
Mat img_load(string address)
{
    Mat img = imread(address, IMREAD_UNCHANGED);
    return img;
}
int getphoto(Mat &frame)
{
    try{
        VideoCapture camera(2);
    }
    catch (...)
    {
        return CAM1_OPEN_ERR;
    }
    VideoCapture camera(2);
    camera.set(CV_CAP_PROP_FRAME_WIDTH,3264);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,2448);
    int i = 5;
    while(i--){
        camera>>frame;
        usleep(1000000);
    }
    frame = frame(Rect(0,0,frame.cols,roi));
    return M_SUCCESS;
}
int getphoto2(Mat &frame)
{
    try{
        VideoCapture camera(3);
    }
    catch (...)
    {
        return CAM2_OPEN_ERR;
    }
    VideoCapture camera(3);
    camera.set(CV_CAP_PROP_FRAME_WIDTH,3264);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,2448);
    int i = 5;
    while(i--){
        camera>>frame;
        usleep(1000000);
    }
    frame = frame(Rect(0,2448-roi,frame.cols,roi));
    return M_SUCCESS;
}
// order camera to get rnt
//2019.11.16 add new method function
int getzx_camera1(vector<struct zhixin> &zx)
{
    int errorcode;
    Mat img;
    errorcode = getphoto(img);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    imwrite("/home/root/test1.jpg", img);
    char s[80];
    time_t now;
    int unixTime = (int)time(&now);
    struct tm *p;
    //now += 28800;
    p = gmtime(&now);
    strftime(s, 80,"%Y%m%d_%H-%M-%S", p);
    string bjtm = s;
    string com = "/media/mmcblk1p1/"+bjtm + "cam1.jpg";
    imwrite(com, img);
    errorcode = image_processing2(img, zx);
    
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    return M_SUCCESS;
}
int getzx_camera2(vector<struct zhixin> &zx)
{
    int errorcode;
    Mat img;
    errorcode = getphoto2(img);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    imwrite("/home/root/test2.jpg", img);
    char s[80];
    time_t now;
    int unixTime = (int)time(&now);
    struct tm *p;
    //now += 28800;
    p = gmtime(&now);
    strftime(s, 80,"%Y%m%d_%H-%M-%S", p);
    string bjtm = s;
    string com = "/media/mmcblk1p1/"+bjtm + "cam2.jpg";
    imwrite(com, img);
    errorcode = image_processing2(img, zx);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    return M_SUCCESS;
}
int resultol_camera1(short &result)
{
    int errorcode;
    vector<struct zhixin>zx2;
    errorcode = getzx_camera1(zx2);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    Mat init_photo = img_load(init_address);
    vector<struct zhixin> zx;
    errorcode = image_processing2(init_photo, zx);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    double result_tmp = calcdistance(zx,zx2);
    if(result_tmp<-100||result_tmp>100)
    {
        errorcode = CAM1_RESULT_ERR;
        return errorcode;
    }
    result = (short)(((int)(abs(result_tmp)*10)) & 0x0fff) | ((result_tmp<0) ? 0x1000:0x0);
    
    return M_SUCCESS;
}
int resultol_camera2(short &result)
{
    int errorcode;
    vector<struct zhixin>zx2;
    errorcode = getzx_camera2(zx2);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    Mat init_photo = img_load(init_address2);
    vector<struct zhixin> zx;
    errorcode = image_processing2(init_photo, zx);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    double result_tmp = calcdistance(zx,zx2);
    if(result_tmp<-100||result_tmp>100)
    {
        errorcode = CAM2_RESULT_ERR;
        return errorcode;
    }
    result = (short)(((int)(abs(result_tmp)*10)) & 0x0fff) | ((result_tmp<0) ? 0x1000:0x0);
    
    return M_SUCCESS;
}
// change the initial image address
int camera_calibrate()
{
    Mat frame;
    int errorcode;
    errorcode = getphoto(frame);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    };
    imwrite(init_address, frame);
    Mat cam;
    cam = imread(init_address, IMREAD_UNCHANGED);
    if(!cam.data)
    {
        errorcode=CAM1_RST_ERR;
        return errorcode;
    }
    return M_SUCCESS;
}
int camera2_calibrate()
{
    Mat frame;
    int errorcode;
    errorcode = getphoto2(frame);
    if(errorcode!=M_SUCCESS)
    {
        return errorcode;
    }
    imwrite(init_address2, frame);
    Mat cam;
    cam = imread(init_address2, IMREAD_UNCHANGED);
    if(!cam.data)
    {
        errorcode=CAM2_RST_ERR;
        return errorcode;
    }
    return M_SUCCESS;
}
int ResetOri(int &ret1, int &ret2)
{
    ret1 = camera_calibrate();
    ret2 = camera2_calibrate();
    return M_SUCCESS;
}
// 11.16 最终输出函数
int RailwayMeasure(short &Offset1,short &Offset2,int &ret1,int &ret2)
{
    ret1 = resultol_camera1(Offset1);
    ret2 = resultol_camera2(Offset2);
    return M_SUCCESS;
}
/*01.16检测模糊度
 返回值为模糊度，值越大越模糊，越小越清晰，范围在0到几十，10以下相对较清晰，经过测试，认为5可以作为衡量是否对焦成功的阈值。
*/


int VideoBlurDetectCam1(cv::Mat &srcimg)

{
    cv::Mat img;
    cv::cvtColor(srcimg, img, COLOR_BGR2GRAY); // 将输入的图片转为灰度图，使用灰度图检测模糊度
 
    //图片每行字节数及高
    int width = img.cols;
    int height = img.rows;
    ushort* sobelTable = new ushort[width*height];
    memset(sobelTable, 0, width*height*sizeof(ushort));
 
    int i, j, mul;
    //指向图像首地址
    uchar* udata = img.data;
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    for (j = 1; j < width - 1; j++)
 
        sobelTable[mul + j] = abs(udata[mul + j - width - 1] + 2 * udata[mul + j - 1] + udata[mul + j - 1 + width] - \
        udata[mul + j + 1 - width] - 2 * udata[mul + j + 1] - udata[mul + j + width + 1]);
 
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    for (j = 1; j < width - 1; j++)
    if (sobelTable[mul + j] < 50 || sobelTable[mul + j] <= sobelTable[mul + j - 1] || \
        sobelTable[mul + j] <= sobelTable[mul + j + 1]) sobelTable[mul + j] = 0;
 
    int totLen = 0;
    int totCount = 1;
 
    uchar suddenThre = 50;
    uchar sameThre = 3;
    //遍历图片
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    {
        for (j = 1; j < width - 1; j++)
        {
            if (sobelTable[mul + j])
            {
                int   count = 0;
                uchar tmpThre = 5;
                uchar max = udata[mul + j] > udata[mul + j - 1] ? 0 : 1;
 
                for (int t = j; t > 0; t--)
                {
                    count++;
                    if (abs(udata[mul + t] - udata[mul + t - 1]) > suddenThre)
                        break;
 
                    if (max && udata[mul + t] > udata[mul + t - 1])
                        break;
 
                    if (!max && udata[mul + t] < udata[mul + t - 1])
                        break;
 
                    int tmp = 0;
                    for (int s = t; s > 0; s--)
                    {
                        if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
                        {
                            tmp++;
                            if (tmp > tmpThre) break;
                        }
                        else break;
                    }
 
                    if (tmp > tmpThre) break;
                }
 
                max = udata[mul + j] > udata[mul + j + 1] ? 0 : 1;
 
                for (int t = j; t < width; t++)
                {
                    count++;
                    if (abs(udata[mul + t] - udata[mul + t + 1]) > suddenThre)
                        break;
 
                    if (max && udata[mul + t] > udata[mul + t + 1])
                        break;
 
                    if (!max && udata[mul + t] < udata[mul + t + 1])
                        break;
 
                    int tmp = 0;
                    for (int s = t; s < width; s++)
                    {
                        if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
                        {
                            tmp++;
                            if (tmp > tmpThre) break;
                        }
                        else break;
                    }
 
                    if (tmp > tmpThre) break;
                }
                count--;
 
                totCount++;
                totLen += count;
            }
        }
    }
    //模糊度
    float result = (float)totLen / totCount;
    delete[] sobelTable;
    sobelTable = NULL;
    line(srcimg, Point(0,80), Point(srcimg.cols,80), Scalar(0,0,255),2);
    line(srcimg, Point(0,160), Point(srcimg.cols,160), Scalar(0,0,255),2);
    //line(srcimg, Point(0,408), Point(srcimg.cols,408), Scalar(0,0,255),2);
    //line(srcimg, Point(0,816), Point(srcimg.cols,816), Scalar(0,0,255),2);
    return result;
}

int VideoBlurDetectCam2(cv::Mat &srcimg)
{
    cv::Mat img;
    cv::cvtColor(srcimg, img, COLOR_BGR2GRAY); // 将输入的图片转为灰度图，使用灰度图检测模糊度
 
    //图片每行字节数及高
    int width = img.cols;
    int height = img.rows;
    ushort* sobelTable = new ushort[width*height];
    memset(sobelTable, 0, width*height*sizeof(ushort));
 
    int i, j, mul;
    //指向图像首地址
    uchar* udata = img.data;
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    for (j = 1; j < width - 1; j++)
 
        sobelTable[mul + j] = abs(udata[mul + j - width - 1] + 2 * udata[mul + j - 1] + udata[mul + j - 1 + width] - \
        udata[mul + j + 1 - width] - 2 * udata[mul + j + 1] - udata[mul + j + width + 1]);
 
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    for (j = 1; j < width - 1; j++)
    if (sobelTable[mul + j] < 50 || sobelTable[mul + j] <= sobelTable[mul + j - 1] || \
        sobelTable[mul + j] <= sobelTable[mul + j + 1]) sobelTable[mul + j] = 0;
 
    int totLen = 0;
    int totCount = 1;
 
    uchar suddenThre = 50;
    uchar sameThre = 3;
    //遍历图片
    for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
    {
        for (j = 1; j < width - 1; j++)
        {
            if (sobelTable[mul + j])
            {
                int   count = 0;
                uchar tmpThre = 5;
                uchar max = udata[mul + j] > udata[mul + j - 1] ? 0 : 1;
 
                for (int t = j; t > 0; t--)
                {
                    count++;
                    if (abs(udata[mul + t] - udata[mul + t - 1]) > suddenThre)
                        break;
 
                    if (max && udata[mul + t] > udata[mul + t - 1])
                        break;
 
                    if (!max && udata[mul + t] < udata[mul + t - 1])
                        break;
 
                    int tmp = 0;
                    for (int s = t; s > 0; s--)
                    {
                        if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
                        {
                            tmp++;
                            if (tmp > tmpThre) break;
                        }
                        else break;
                    }
 
                    if (tmp > tmpThre) break;
                }
 
                max = udata[mul + j] > udata[mul + j + 1] ? 0 : 1;
 
                for (int t = j; t < width; t++)
                {
                    count++;
                    if (abs(udata[mul + t] - udata[mul + t + 1]) > suddenThre)
                        break;
 
                    if (max && udata[mul + t] > udata[mul + t + 1])
                        break;
 
                    if (!max && udata[mul + t] < udata[mul + t + 1])
                        break;
 
                    int tmp = 0;
                    for (int s = t; s < width; s++)
                    {
                        if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
                        {
                            tmp++;
                            if (tmp > tmpThre) break;
                        }
                        else break;
                    }
 
                    if (tmp > tmpThre) break;
                }
                count--;
 
                totCount++;
                totLen += count;
            }
        }
    }
    //模糊度
    float result = (float)totLen / totCount;
    delete[] sobelTable;
    sobelTable = NULL;
    //line(srcimg, Point(0,2040), Point(srcimg.cols,2040), Scalar(0,0,255),2);
    //line(srcimg, Point(0,1632), Point(srcimg.cols,1632), Scalar(0,0,255),2);
    line(srcimg, Point(0,400), Point(srcimg.cols,400), Scalar(0,0,255),2);
    line(srcimg, Point(0,320), Point(srcimg.cols,320), Scalar(0,0,255),2);
    return result;
}
//2020.04.16
Mat PerfectReflectionAlgorithm(Mat src)
{
    int row = src.rows;
    int col = src.cols;
    Mat dst(row, col, CV_8UC3);
    int HistRGB[767] = { 0 };
    int MaxVal = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
            int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
            HistRGB[sum]++;
        }
    }
    int Threshold = 0;
    int sum = 0;
    for (int i = 766; i >= 0; i--) {
        sum += HistRGB[i];
        if (sum > row * col * 0.1) {
            Threshold = i;
            break;
        }
    }
    int AvgB = 0;
    int AvgG = 0;
    int AvgR = 0;
    int cnt = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
            if (sumP > Threshold) {
                AvgB += src.at<Vec3b>(i, j)[0];
                AvgG += src.at<Vec3b>(i, j)[1];
                AvgR += src.at<Vec3b>(i, j)[2];
                cnt++;
            }
        }
    }
    AvgB /= cnt;
    AvgG /= cnt;
    AvgR /= cnt;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
            int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
            int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
            if (Red > 255) {
                Red = 255;
            }
            else if (Red < 0) {
                Red = 0;
            }
            if (Green > 255) {
                Green = 255;
            }
            else if (Green < 0) {
                Green = 0;
            }
            if (Blue > 255) {
                Blue = 255;
            }
            else if (Blue < 0) {
                Blue = 0;
            }
            dst.at<Vec3b>(i, j)[0] = Blue;
            dst.at<Vec3b>(i, j)[1] = Green;
            dst.at<Vec3b>(i, j)[2] = Red;
        }
    }
    return dst;
}
// 2020.04.22 获得标尺斜率函数
float getrulerslope(vector<struct zhixin> zx)
{
    vector<Point2f> uppoints;
    for(int i = 0;i<zx.size();i++)
    {
        if (zx[i].isupside==true)
        {
            uppoints.push_back(zx[i].position);
        }
    }
    //用上刻度的质心进行直线拟合
    Vec4i rulerfitline;
    fitLine(uppoints, rulerfitline, DIST_L2, 0, 0.01, 0.01);
    float k;
    if(rulerfitline[0]==0)
    {
        k = 1;
    }
    else
    {
        k = (double)rulerfitline[1]/(double)rulerfitline[0];
    }
    return k;
}
//输入两组一一对应的点，得到两组点连线的斜率均值
float avgslope(vector<Point2f> ruler1, vector<Point2f> ruler2)
{
    vector<float> k;
    for(int i = 0; i<ruler1.size();i++)
    {
        k.push_back((double)(ruler2[i].y-ruler1[i].y)/(double)(ruler2[i].x-ruler1[i].x));
    }
    float sum=0;
    int num = k.size();
    for(int i = 0;i<k.size();i++)
    {
        sum+=k[i];
    }
    float avgresult = sum/num;
    return avgresult;
}
