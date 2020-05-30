#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void kmeans_data_demo();
void kmeans_image_demo();
void kmeans_background_replace();
void kmeans_color_card();
int main(int argc, char** argv) {
	// kmeans_data_demo();
	// kmeans_image_demo();
	kmeans_background_replace();
	// kmeans_color_card();
	return 0;

	waitKey(0);
	return 0;
}

void kmeans_data_demo() {
	Mat img(500, 500, CV_8UC3);
	RNG rng(12345);

	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(255, 0, 0),
	};

	int numCluster = 2;
	int sampleCount = rng.uniform(5, 500);
	Mat points(sampleCount, 1, CV_32FC2);

	// ���������
	for (int k = 0; k < numCluster; k++) {
		Point center;
		center.x = rng.uniform(0, img.cols);
		center.y = rng.uniform(0, img.rows);
		Mat pointChunk = points.rowRange(k*sampleCount / numCluster,
			k == numCluster - 1 ? sampleCount : (k + 1)*sampleCount / numCluster);
		rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
	}
	randShuffle(points, 1, &rng);

	// ʹ��KMeans
	Mat labels;
	Mat centers;
	kmeans(points, numCluster, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, centers);


	// �ò�ͬ��ɫ��ʾ����
	img = Scalar::all(255);
	for (int i = 0; i < sampleCount; i++) {
		int index = labels.at<int>(i);
		Point p = points.at<Point2f>(i);
		circle(img, p, 2, colorTab[index], -1, 8);
	}

	// ÿ�����������������Բ
	for (int i = 0; i < centers.rows; i++) {
		int x = centers.at<float>(i, 0);
		int y = centers.at<float>(i, 1);
		printf("c.x= %d, c.y=%d\n", x, y);
		circle(img, Point(x, y), 40, colorTab[i], 1, LINE_AA);
	}

	imshow("KMeans-Data-Demo", img);
	waitKey(0);
}
void kmeans_image_demo() {
	Mat src = imread("10.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	Vec3b colorTab[] = {
		Vec3b(0, 0, 255),
		Vec3b(0, 255, 0),
		Vec3b(255, 0, 0),
		Vec3b(0, 255, 255),
		Vec3b(255, 0, 255)
	};

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// ��ʼ������
	int sampleCount = width*height;
	int clusterCount = 3;
	Mat labels;
	Mat centers;

	// RGB ����ת������������
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// ����K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	// ��ʾͼ��ָ���
	int index = 0;
	Mat result = Mat::zeros(src.size(), src.type());
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			result.at<Vec3b>(row, col) = colorTab[label];
		}
	}

	imshow("KMeans-image-Demo", result);
	waitKey(0);
}
void kmeans_background_replace() {
	Mat src = imread("D:/projects/opencv_tutorial/data/images/toux.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// ��ʼ������
	int sampleCount = width*height;
	int clusterCount = 3;
	Mat labels;
	Mat centers;

	// RGB ����ת������������
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// ����K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	// ����mask
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = labels.at<int>(0, 0);
	labels = labels.reshape(1, height);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int c = labels.at<int>(row, col);
			if (c == index) {
				mask.at<uchar>(row, col) = 255;
			}
		}
	}
	imshow("mask", mask);

	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(mask, mask, se);

	// ���ɸ�˹Ȩ��
	GaussianBlur(mask, mask, Size(5, 5), 0);
	imshow("mask-blur", mask);

	// ���ڸ�˹Ȩ��ͼ���ں�
	Mat result = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float w1 = mask.at<uchar>(row, col) / 255.0;
			Vec3b bgr = src.at<Vec3b>(row, col);
			bgr[0] = w1 * 255.0 + bgr[0] * (1.0 - w1);
			bgr[1] = w1 * 0 + bgr[1] * (1.0 - w1);
			bgr[2] = w1 * 255.0 + bgr[2] * (1.0 - w1);
			result.at<Vec3b>(row, col) = bgr;
		}
	}
	imshow("background-replacement-demo", result);
	waitKey(0);
}
void kmeans_color_card() {
	Mat src = imread("D:/images/master.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// ��ʼ������
	int sampleCount = width*height;
	int clusterCount = 4;
	Mat labels;
	Mat centers;

	// RGB ����ת������������
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// ����K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	Mat card = Mat::zeros(Size(width, 50), CV_8UC3);
	vector<float> clusters(clusterCount);

	// ����ɫ������
	for (int i = 0; i < labels.rows; i++) {
		clusters[labels.at<int>(i, 0)]++;
	}

	for (int i = 0; i < clusters.size(); i++) {
		clusters[i] = clusters[i] / sampleCount;
	}
	int x_offset = 0;

	// ����ɫ��
	for (int x = 0; x < clusterCount; x++) {
		Rect rect;
		rect.x = x_offset;
		rect.y = 0;
		rect.height = 50;
		rect.width = round(clusters[x] * width);
		x_offset += rect.width;
		int b = centers.at<float>(x, 0);
		int g = centers.at<float>(x, 1);
		int r = centers.at<float>(x, 2);
		rectangle(card, rect, Scalar(b, g, r), -1, 8, 0);
	}

	imshow("Image Color Card", card);
	waitKey(0);
}