#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat tpl;
void sort_box(vector<Rect> &boxes);
void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect);
int main(int argc, char** argv) {
	Mat src = imread("7.jpg");
	if (src.empty()) {
		printf("could not load image file...");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// ͼ���ֵ��
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);

	// ����ṹԪ��
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, se);

	// ��������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;
	findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	int height = src.rows;
	for (size_t t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);
		double area = contourArea(contours[t]);
		if (rect.height > (height / 2)) {
			continue;
		}
		if (area < 150) {
			continue;
		}
		rects.push_back(rect);
		// rectangle(src, rect, Scalar(0, 0, 255), 2, 8, 0);
		// drawContours(src, contours, t, Scalar(0, 0, 255), 2, 8);
	}
	sort_box(rects);
	tpl = binary(rects[1]);

	// for (int i = 0; i < rects.size(); i++) {
	// 	putText(src, format("%d", i), rects[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1, 8);
	// }
	vector<Rect> defects;
	detect_defect(binary, rects, defects);

	for (int i = 0; i < defects.size(); i++) {
		rectangle(src, defects[i], Scalar(0, 0, 255), 2, 8, 0);
		putText(src, "bad", defects[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1, 8);
	}
	imshow("detect result", src);
	imwrite("D:/detection_result.png", src);
	waitKey(0);
	return 0;
}

void sort_box(vector<Rect> &boxes) {
	int size = boxes.size();
	for (int i = 0; i < size - 1; i++) {
		for (int j = i; j < size; j++) {
			int x = boxes[j].x;
			int y = boxes[j].y;
			if (y < boxes[i].y) {
				Rect temp = boxes[i];
				boxes[i] = boxes[j];
				boxes[j] = temp;
			}
		}
	}
}

void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect) {
	int h = tpl.rows;
	int w = tpl.cols;
	int size = rects.size();
	for (int i = 0; i < size; i++) {
		// ����diff
		Mat roi = binary(rects[i]);
		resize(roi, roi, tpl.size());
		Mat mask;
		subtract(tpl, roi, mask);
		Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		morphologyEx(mask, mask, MORPH_OPEN, se);
		threshold(mask, mask, 0, 255, THRESH_BINARY);
		imshow("mask", mask);
		waitKey(0);
		
		// ����diff����ȱ�ݣ���ֵ��
		int count = 0;
		for (int row = 0; row < h; row++) {
			for (int col = 0; col < w; col++) {
				int pv = mask.at<uchar>(row, col);
				if (pv == 255) {
					count++;
				}
			}
		}

		// ���һ�����ؿ�
		int mh = mask.rows + 2;
		int mw = mask.cols + 2;
		Mat m1 = Mat::zeros(Size(mw, mh), mask.type());
		Rect mroi;
		mroi.x = 1;
		mroi.y = 1;
		mroi.height = mask.rows;
		mroi.width = mask.cols;
		mask.copyTo(m1(mroi));

		// ��������
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(m1, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
		bool find = false;
		for (size_t t = 0; t < contours.size(); t++) {
			Rect rect = boundingRect(contours[t]);
			float ratio = (float)rect.width / ((float)rect.height);
			if (ratio > 4.0 && (rect.y < 5 || (m1.rows - (rect.height + rect.y)) < 10)) {
				continue;
			}
			double area = contourArea(contours[t]);
			if (area > 10) {
				printf("ratio : %.2f, area : %.2f \n", ratio, area);
				find = true;
			}
		}

		if (count > 50 && find) {
			printf("count : %d \n", count);
			defect.push_back(rects[i]);
		}
	}
}