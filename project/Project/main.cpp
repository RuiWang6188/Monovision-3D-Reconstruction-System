#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
using namespace cv;
using namespace std;

#define PI 3.1415926

void calRealPoint(std::vector<std::vector<cv::Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, float squaresize)
{
	//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
	squaresize = 30;
	std::vector<cv::Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0); 
			//7 7 20 3mm?
			//3,4 => (3*3mm-(7/2*3mm),4*3mm-(7/2*3mm),0)
			imgpoint.push_back(cv::Point3f((float)colIndex * squaresize - (boardwidth / 2 * squaresize), (float)rowIndex * squaresize - (boardheight / 2 * squaresize), 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

//像素位置、内参、R、T==》世界坐标
Point3f getWorldPoints(Point2f& inPoints, Mat& rvec, Mat& tvec, Mat& cameraMatrix)
{
	//initialize parameter
	Mat rotationMatrix;//3*3 
	Rodrigues(rvec, rotationMatrix);
	double zConst = 0;//实际坐标系的距离，若工作平面与相机距离固定可设置为0
	double s;

	//获取图像坐标
	cv::Mat imagePoint = (Mat_<double>(3, 1) << double(inPoints.x), double(inPoints.y), 1);
	// cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	// imagePoint.at<double>(0, 0) = inPoints.x;
	// imagePoint.at<double>(1, 0) = inPoints.y;

	//计算比例参数S
	cv::Mat tempMat, tempMat2;
	tempMat = rotationMatrix.inv() * cameraMatrix.inv() * imagePoint;
	tempMat2 = rotationMatrix.inv() * tvec;
	s = zConst + tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);

	//计算世界坐标
	Mat wcPoint = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePoint - tvec);
	Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
	return worldPoint;
}

//像素位置、内参、R、T==》》相机坐标
Point3f getCameraPoints(Point2f& inPoints, Mat& cameraMatrix)
{
	//获取图像坐标
	cv::Mat imagePoint = (Mat_<double>(3, 1) << double(inPoints.x), double(inPoints.y), 1);
	// cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	// imagePoint.at<double>(0, 0) = inPoints.x;
	// imagePoint.at<double>(1, 0) = inPoints.y;

	//计算世界坐标
	Mat caPoint = cameraMatrix.inv() * imagePoint;
	Point3f cameraPoint(caPoint.at<double>(0, 0), caPoint.at<double>(1, 0), caPoint.at<double>(2, 0));
	return cameraPoint;
}

//Ax+by+cz=D
void cvFitPlane(const CvMat* points, float* plane) {
	// Estimate geometric centroid.
	int nrows = points->rows;
	int ncols = points->cols;
	int type = points->type;
	CvMat* centroid = cvCreateMat(1, ncols, type);
	cvSet(centroid, cvScalar(0));
	for (int c = 0; c < ncols; c++) {
		for (int r = 0; r < nrows; r++)
		{
			centroid->data.fl[c] += points->data.fl[ncols * r + c];
		}
		centroid->data.fl[c] /= nrows;
	}
	// Subtract geometric centroid from each point.
	CvMat* points2 = cvCreateMat(nrows, ncols, type);
	for (int r = 0; r < nrows; r++)
		for (int c = 0; c < ncols; c++)
			points2->data.fl[ncols * r + c] = points->data.fl[ncols * r + c] - centroid->data.fl[c];
	// Evaluate SVD of covariance matrix.
	CvMat* A = cvCreateMat(ncols, ncols, type);
	CvMat* W = cvCreateMat(ncols, ncols, type);
	CvMat* V = cvCreateMat(ncols, ncols, type);
	cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
	cvSVD(A, W, NULL, V, CV_SVD_V_T);
	// Assign plane coefficients by singular vector corresponding to smallest singular value.
	plane[ncols] = 0;
	for (int c = 0; c < ncols; c++) {
		plane[c] = V->data.fl[ncols * (ncols - 1) + c];
		plane[ncols] += plane[c] * centroid->data.fl[c];
	}
	// Release allocated resources.
	cvReleaseMat(&centroid);
	cvReleaseMat(&points2);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&V);
}

cv::Point3f myRotate(cv::Point3f p, cv::Vec6f line, int k) {
	//自己猜测参数？
	Mat res = (Mat_<double>(3, 1) << p.x - line[3], p.y - line[4]-0.05, p.z - line[5]);
	Mat rotationMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(k * PI / 160), sin(k * PI / 160), 0, -sin(k * PI / 160), cos(k * PI / 160));
	res= rotationMatrix * res;
	cv::Point3f final(res.at<double>(0,0), res.at<double>(1, 0), res.at<double>(2, 0));
	return final;
}


//主函数
int main()
{
	//检测标定板
	cout << "---------------------Step1:标定-------------------" << endl;
	cv::Mat rgbImage, grayImage;
	std::vector<cv::Point2f> corner;
	std::vector<std::vector<cv::Point2f>> imagePoint;
	for (int i = 1; i <= 20; i++)
	{
		string path = "./img/calib/" + to_string(i) + ".bmp";
		rgbImage = cv::imread(path, CV_LOAD_IMAGE_COLOR);

		cout << "Read image success: " << path << endl;

		cv::cvtColor(rgbImage, grayImage, CV_BGR2GRAY);

		bool isFind;
		isFind = findChessboardCorners(grayImage, Size(8, 6), corner, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		cout << "Find corner!" << endl;

		//isFind = findCirclesGrid(grayImage, cv::Size(7, 7), corner);
		//cout << "Find grid!" << endl;

		if (isFind)
		{
			//cornerSubPix(grayImage, corner, cv::Size(7, 7), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(rgbImage, cv::Size(8, 6), corner, isFind);
			imagePoint.push_back(corner);

			//if (i == 10) {
			//	cv::imshow("Image", rgbImage);
			//	cv::waitKey(0);
			//}
		}
	}

	//标准图用于投影变换
	std::vector<std::vector<cv::Point3f>> objRealPoint;
	calRealPoint(objRealPoint, 8, 6, 20, 3);

	//标定
	cv::Mat cameraMatrix, distCoeff;
	vector<Mat> rvecsMat;
	vector<Mat> tvecsMat;
	float rms = calibrateCamera(objRealPoint, imagePoint, cv::Size(rgbImage.cols, rgbImage.rows), cameraMatrix, distCoeff, rvecsMat, tvecsMat, CV_CALIB_FIX_K3);
	cout << "Find 20 camera parameter matrix!" << endl;

	

	cout << "---------------------Step2:拟合激光平面-------------------" << endl;

	std::vector<cv::Point3f> Points3d_19;
	//第19张标定图激光图==》》三维坐标(Z坐标为0)
	//cameraMatrix, distCoeff, rvecsMat[18], tvecsMat[18]
	Mat lightline_19 = imread("./img/calib/19-1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	//cout << "Original Image 19-1: " << lightline_19 << endl;

	threshold(lightline_19, lightline_19, 80, 255, THRESH_BINARY);

	//cv::namedWindow("Image", cv::WINDOW_NORMAL);
	//imshow("Image", lightline_19);
	//cv::waitKey(0);

	for (size_t i = 0; i < lightline_19.cols; i++)
	{
		int sum = 0; int num = 0;
		for (size_t j = 0; j < lightline_19.rows; j++)
		{
			if (lightline_19.at<uchar>(j, i) == 255)
			{
				sum += j;
				num++;
			}
		}
		if (num == 0)
			continue;
		Point2f temp1 = Point2f(i, 1.0 * sum / num);	
		Points3d_19.push_back(getCameraPoints(temp1, cameraMatrix));
	}

	cout << "Point3d_19-1 Find!" << endl;

	std::vector<cv::Point3f> Points3d_20;
	//第20张标定图激光图==》》三维坐标(Z坐标为0)需要转到19张图坐标系下
	//cameraMatrix, distCoeff, rvecsMat[19], tvecsMat[19]
	Mat lightline_20 = imread("./img/calib/20-1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	threshold(lightline_20, lightline_20, 80, 255, THRESH_BINARY);	//MOD

	//cv::namedWindow("Image", cv::WINDOW_NORMAL);
	//imshow("Image", lightline_20);
	//cv::waitKey(0);

	for (size_t i = 0; i < lightline_20.cols; i++)
	{
		int sum = 0; int num = 0;
		for (size_t j = 0; j < lightline_20.rows; j++)
		{
			if (lightline_20.at<uchar>(j, i) == 255)
			{
				sum += j;
				num++;
			}
		}
		if (num == 0)
			continue;
		Point2f temp2 = Point2f(i, 1.0 * sum / num);			// MOD
		Points3d_20.push_back(getCameraPoints(temp2, cameraMatrix));
	}

	cout << "Point3d_20-1 Find!" << endl;

	//拟合激光平面

	//点云1
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针）
	(*cloud).points.resize(Points3d_19.size() + Points3d_20.size());


	CvMat* points_mat = cvCreateMat(Points3d_19.size() + Points3d_20.size(), 3, CV_32FC1);//定义用来存储需要拟合点的矩阵 
	for (int i = 0; i < Points3d_19.size(); ++i)
	{
		(*cloud).points[i].x = Points3d_19[i].x;
		(*cloud).points[i].y = Points3d_19[i].y;
		(*cloud).points[i].z = Points3d_19[i].z;
		points_mat->data.fl[i * 3 + 0] = Points3d_19[i].x;//矩阵的值进行初始化   X的坐标值
		points_mat->data.fl[i * 3 + 1] = Points3d_19[i].y;//  Y的坐标值
		points_mat->data.fl[i * 3 + 2] = Points3d_19[i].z;//  Z的坐标值</span>
	}
	for (int i = 0; i < Points3d_20.size(); ++i)
	{
		(*cloud).points[i + Points3d_19.size()].x = Points3d_20[i].x;
		(*cloud).points[i + Points3d_19.size()].y = Points3d_20[i].y;
		(*cloud).points[i + Points3d_19.size()].z = Points3d_20[i].z;
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 0] = Points3d_20[i].x;//矩阵的值进行初始化   X的坐标值
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 1] = Points3d_20[i].y;//  Y的坐标值
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 2] = Points3d_20[i].z;//  Z的坐标值</span>
	}


	cout << "points_mat: " << points_mat->rows << " * " << points_mat->cols << endl;


	float line_plane[4] = { 0 };//定义用来储存平面参数的数组 
	cvFitPlane(points_mat, line_plane);//拟合平面方程 

	cout << "Plane Done!:	" << line_plane[0] << "x + " << line_plane[1] << "y + " << line_plane[2] << "z = " << line_plane[3] << endl;



	//cout << "---------------------Step3:确定履带位移-------------------" << endl;		//转盘未用到该模块
	//确定履带位移：根据内参+像素点 => 计算RT
	/*std::vector<cv::Point2f> corner_1, corner_20;

	Mat caltab_at_position_1 = imread("D:/Class/Level_Four/IVIA/Experiment/project/image/img5/calib-speed/Basler_daA2500-14uc__40031429__20201109_195140195_44.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	bool isFind_1 = findCirclesGrid(caltab_at_position_1, cv::Size(7, 7), corner_1); 
	//drawChessboardCorners(caltab_at_position_1, cv::Size(7, 7), corner_1, isFind_1);

	Mat caltab_at_position_20 = imread("D:/Class/Level_Four/IVIA/Experiment/project/image/img5/calib-speed/Basler_daA2500-14uc__40031429__20201109_195140195_63.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	bool isFind_20 = findCirclesGrid(caltab_at_position_20, cv::Size(7, 7), corner_20);
	//drawChessboardCorners(caltab_at_position_20, cv::Size(7, 7), corner_20, isFind_20);

	/*std::vector<std::vector<cv::Point3f>> objRealPoint2;
	calRealPoint(objRealPoint2, 7, 7, 2, 3);
	std::vector<std::vector<cv::Point2f>> imagePoint2;
	imagePoint2.push_back(corner_1);
	imagePoint2.push_back(corner_20);

	vector<Mat> rvecsMat2;
	rvecsMat2.resize(2);
	vector<Mat> tvecsMat2;
	tvecsMat2.resize(2);
	solvePnP(objRealPoint2[0], imagePoint2[0], cameraMatrix, distCoeff, rvecsMat2[0], tvecsMat2[0], false, SOLVEPNP_DLS);
	solvePnP(objRealPoint2[1], imagePoint2[1], cameraMatrix, distCoeff, rvecsMat2[1], tvecsMat2[1], false, SOLVEPNP_DLS);

	Mat Point3d_mat = (Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	Mat rotationMatrix1;//3*3
	Rodrigues(rvecsMat2[0], rotationMatrix1);
	Mat rotationMatrix20;//3*3
	Rodrigues(rvecsMat2[1], rotationMatrix20);
	Mat rotationMatrix19;//3*3
	Rodrigues(rvecsMat[18], rotationMatrix19);//标杆

	Mat Point3d_1to19_mat = rotationMatrix19 * rotationMatrix1.inv() * (Point3d_mat - tvecsMat2[0]) + tvecsMat[18];
	Mat Point3d_20to19_mat = rotationMatrix19 * rotationMatrix20.inv() * (Point3d_mat - tvecsMat2[1]) + tvecsMat[18];
	
	cv::Point3f c1 = getCameraPoints(corner_1[0], cameraMatrix);
	Mat Point3d_1 = (Mat_<double>(3, 1) << c1.x, c1.y, c1.z); 
	cv::Point3f c20 = getCameraPoints(corner_20[0], cameraMatrix);
	Mat Point3d_20 = (Mat_<double>(3, 1) << c20.x, c20.y, c20.z);
	
	Mat move_steps = Point3d_20 - Point3d_1;//1-20移动距离
	Mat move_step = move_steps / 19;//单步移动距离

	Mat move_step = (Mat_<double>(3, 1) << line_plane[0], line_plane[1], line_plane[2]) / 800;
	cout << "Move_step compute success!" << endl;
	cout << "move_step: " << move_step << endl;
	*/

	cout << "---------------------Step3:计算转盘中心轴-------------------" << endl;

	Mat center = imread("./img/calib/center.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	threshold(center, center, 80, 255, THRESH_BINARY);
	std::vector<cv::Point3f> centerPoint_all;
	for (size_t i = center.cols/3; i < center.cols*2/3; i++)
	{
		int sum = 0; int num = 0;
		for (size_t j = 0; j < center.rows; j++)
		{
			if (center.at<uchar>(j, i) == 255)
			{
				sum += j;
				num++;
			}
		}
		if (num == 0)
			continue;
		Point2f temp3 = Point2f(i, 1.0 * sum / num);		
		cv::Point3f centerPoint = getCameraPoints(temp3, cameraMatrix);
		centerPoint.z = (line_plane[3] - line_plane[0] * centerPoint.x - line_plane[1] * centerPoint.y) / line_plane[2];
		centerPoint_all.push_back(centerPoint);
	}
	cv::Vec6f centerLine;
	cv::fitLine(centerPoint_all, centerLine, cv::DIST_L2, 0, 0.01, 0.01);
	

	cout << "---------------------Step4:计算物体点云-------------------" << endl;

	//计算
	std::vector<cv::Point3f> Points3d_all;
	for (size_t k = 1; k <= 320; k++)
	{
		
		string path = "./img/mug/Basler_daA2500-14uc__40031429__20201111_171829709_" + to_string(k) + ".bmp";
		//string path = "img/continous/dragon.bmp";

		Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

		//cout << "Read Image Success:" << path << endl;

		threshold(image, image, 80, 255, THRESH_BINARY);

		for (size_t i = 0; i < image.cols; i++)
		{
			int sum = 0; int num = 0;
			for (size_t j = 0; j < image.rows; j++)
			{
				if (image.at<uchar>(j, i) == 255)
				{
					sum += j;
					num++;
				}
			}
			if (num == 0)
				continue;
			Point2f temp3 = Point2f(i, 1.0 * sum / num);		//MOD
			cv::Point3f Points3d = getCameraPoints(temp3, cameraMatrix);
			//cv::Point3f Points3d = getWorldPoints(temp3, rvecsMat[18], tvecsMat[18], cameraMatrix);
			Points3d.z = (line_plane[3] - line_plane[0] * Points3d.x - line_plane[1] * Points3d.y) / line_plane[2];
			//Points3d.z = 0;
			//Points3d += Point3f((k - 1) * move_step.at<double>(0, 0), (k - 1) * move_step.at<double>(1, 0), (k - 1) * move_step.at<double>(2, 0));
			cv::Point3f Points3d_final = myRotate(Points3d, centerLine, k);
			Points3d_all.push_back(Points3d_final);

			
		}
		//cout << "The size of the point cloud:" << Points3d_all.size() << endl;
		//imshow("image", image);
		//waitKey(10);
	}
	cout << "Point cloud computation is done!" << endl;

	//显示
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针）
	int size = (*cloud).points.size();
	(*cloud).points.resize(size+Points3d_all.size()+100);
	for (size_t i = 0; i < Points3d_all.size(); i++)
	{
		(*cloud).points[i+size].x = Points3d_all[i].x;
		(*cloud).points[i+size].y = Points3d_all[i].y;
		(*cloud).points[i+size].z = Points3d_all[i].z;
	}


	for (size_t i = 0; i < 100; i++)
	{
		(*cloud).points[i + size + Points3d_all.size()].x = i * 1.0 / 800 * centerLine[0];
		(*cloud).points[i + size + Points3d_all.size()].y = i * 1.0 / 800 * centerLine[1];
		(*cloud).points[i + size + Points3d_all.size()].z = i * 1.0 / 800 * centerLine[2];
	}
	pcl::io::savePLYFileASCII("mug.ply", *cloud);
	pcl::visualization::CloudViewer viewer("ply viewer");
	viewer.showCloud(cloud);
	while (!viewer.wasStopped())
	{
	}
}