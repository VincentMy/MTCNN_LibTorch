//#include <torch/script.h>
#include <iostream>
#include <memory>

#include "MTCNN.h"

int main(int argc, char* argv[]) {
	std::cout << "out" << std::endl;
	
	string model_path = "D:/WorkSpaces/C++/mtcnn_c_test_copy/";
	
	string pnet_weight_path = model_path + "pnet.pt";
	string rnet_weight_path = model_path + "rnet.pt";
	string onet_weight_path = model_path + "onet.pt";
	TAlgParam alg_param;
	alg_param.min_face = 40;
	alg_param.scale_factor = 0.79;
	alg_param.cls_thre[0] = 0.6;
	alg_param.cls_thre[1] = 0.7;
	alg_param.cls_thre[2] = 0.7;

	TModelParam modelParam;
	modelParam.alg_param = alg_param;
	modelParam.model_path = { pnet_weight_path, rnet_weight_path, onet_weight_path };
	modelParam.mean_value = { {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5} };
	modelParam.scale_factor = { 1.0f, 1.0f, 1.0f };

	MTCNN mt;
	mt.InitDetector(&modelParam);
	
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while (cv::waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			std::cout << "Done processing !!" << endl;
			break;
		}
		//opencv中矩形rect
		vector<cv::Rect> outFaces;
		LOGI("warm up ...");
		for (int i = 0; i < 5; i++) {
			mt.DetectFace(frame, outFaces);
		}
		LOGI("DetectFace ...");
		mt.DetectFace(frame, outFaces);
		//LOGI(" cost: {}");

		for (auto& i : outFaces)
			cv::rectangle(frame, i, { 0,255,0 }, 2);
		cv::imshow("result", frame);

	}
	cap.release();
	return 0;
	
	/*
	string img_path = model_path + "faces2.jpg";
	cv::Mat src = cv::imread(img_path);
	if (src.empty())
	{
		LOGE("can not load image");
		return -1;
	}
	//opencv中矩形rect
	vector<cv::Rect> outFaces;
	LOGI("warm up ...");
	for (int i = 0; i < 5; i++) {
		mt.DetectFace(src, outFaces);
	}
	LOGI("DetectFace ...");
	mt.DetectFace(src, outFaces);
	//LOGI(" cost: {}");

	for (auto& i : outFaces)
		cv::rectangle(src, i, { 0,255,0 }, 2);

	cv::imshow("result", src);
	cv::waitKey(0);
	cv::imwrite("res2.jpg", src);
	//LEAVE_FUNC;
	return 0;
	*/
	//cv::namedWindow("input", cv::WINDOW_AUTOSIZE);
	//cv::imshow("input", src);
	//cv::waitKey(0);
	//cv::destroyAllWindows();	
}