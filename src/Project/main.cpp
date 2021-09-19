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
			imgpoint.push_back(cv::Point3f((float)colIndex * squaresize - (boardwidth / 2 * squaresize), (float)rowIndex * squaresize - (boardheight / 2 * squaresize), 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

// Convert pixel coordinate to world coordinate
Point3f getWorldPoints(Point2f& inPoints, Mat& rvec, Mat& tvec, Mat& cameraMatrix)
{
	//initialize parameter
	Mat rotationMatrix; //3*3 
	Rodrigues(rvec, rotationMatrix);
	double zConst = 0;// Since the work space, the laser scanner and the camera is fixed，it can be set to 0
	double s;

	// get pixel coordinate
	cv::Mat imagePoint = (Mat_<double>(3, 1) << double(inPoints.x), double(inPoints.y), 1);
	// cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	// imagePoint.at<double>(0, 0) = inPoints.x;
	// imagePoint.at<double>(1, 0) = inPoints.y;

	// mompute scale parameter S
	cv::Mat tempMat, tempMat2;
	tempMat = rotationMatrix.inv() * cameraMatrix.inv() * imagePoint;
	tempMat2 = rotationMatrix.inv() * tvec;
	s = zConst + tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);

	// compute corresponding world coordinate
	Mat wcPoint = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePoint - tvec);
	Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
	return worldPoint;
}

// convert pixel coordinate to camera coordinate
Point3f getCameraPoints(Point2f& inPoints, Mat& cameraMatrix)
{
	// get pixel coordinate
	cv::Mat imagePoint = (Mat_<double>(3, 1) << double(inPoints.x), double(inPoints.y), 1);
	// cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	// imagePoint.at<double>(0, 0) = inPoints.x;
	// imagePoint.at<double>(1, 0) = inPoints.y;

	// comput corresponding camera coodinate
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
	Mat res = (Mat_<double>(3, 1) << p.x - line[3], p.y - line[4]-0.05, p.z - line[5]);
	Mat rotationMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(k * PI / 160), sin(k * PI / 160), 0, -sin(k * PI / 160), cos(k * PI / 160));
	res= rotationMatrix * res;
	cv::Point3f final(res.at<double>(0,0), res.at<double>(1, 0), res.at<double>(2, 0));
	return final;
}


int main()
{
	// camera calibration
	cout << "---------------------Step1: Camera Calibration-------------------" << endl;
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

	//standard graph are used for projection transformations
	std::vector<std::vector<cv::Point3f>> objRealPoint;
	calRealPoint(objRealPoint, 8, 6, 20, 3);

	//calibration
	cv::Mat cameraMatrix, distCoeff;
	vector<Mat> rvecsMat;
	vector<Mat> tvecsMat;
	float rms = calibrateCamera(objRealPoint, imagePoint, cv::Size(rgbImage.cols, rgbImage.rows), cameraMatrix, distCoeff, rvecsMat, tvecsMat, CV_CALIB_FIX_K3);
	cout << "Find 20 camera parameter matrix!" << endl;

	

	cout << "---------------------Step2:Fit Laser Plane-------------------" << endl;

	std::vector<cv::Point3f> Points3d_19;
	// get the camera coorindate of the 19th laser calibration image 
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
	// the 20th laser calibration image should be converted to the coordinate of the 19th image
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

	// get the laser plane

	// point cloud of laser plane
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); 
	(*cloud).points.resize(Points3d_19.size() + Points3d_20.size());


	CvMat* points_mat = cvCreateMat(Points3d_19.size() + Points3d_20.size(), 3, CV_32FC1); // the matrix is used to store the points to be fitted 
	for (int i = 0; i < Points3d_19.size(); ++i)
	{
		(*cloud).points[i].x = Points3d_19[i].x;
		(*cloud).points[i].y = Points3d_19[i].y;
		(*cloud).points[i].z = Points3d_19[i].z;
		points_mat->data.fl[i * 3 + 0] = Points3d_19[i].x;
		points_mat->data.fl[i * 3 + 1] = Points3d_19[i].y;
		points_mat->data.fl[i * 3 + 2] = Points3d_19[i].z;
	}
	for (int i = 0; i < Points3d_20.size(); ++i)
	{
		(*cloud).points[i + Points3d_19.size()].x = Points3d_20[i].x;
		(*cloud).points[i + Points3d_19.size()].y = Points3d_20[i].y;
		(*cloud).points[i + Points3d_19.size()].z = Points3d_20[i].z;
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 0] = Points3d_20[i].x;
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 1] = Points3d_20[i].y;
		points_mat->data.fl[Points3d_19.size() * 3 + i * 3 + 2] = Points3d_20[i].z;
	}


	cout << "points_mat: " << points_mat->rows << " * " << points_mat->cols << endl;


	float line_plane[4] = { 0 };	// the parameter of the plane equation
	cvFitPlane(points_mat, line_plane);	// fit plane 

	cout << "Plane Done!:	" << line_plane[0] << "x + " << line_plane[1] << "y + " << line_plane[2] << "z = " << line_plane[3] << endl;

	cout << "---------------------Step3: Compute Rotation Axis-------------------" << endl;

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
	

	cout << "---------------------Step4: Compute and Merge Point Cloud-------------------" << endl;

	// Computation
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

	//Display
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