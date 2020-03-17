#pragma once

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <spdhelper.hpp>
using namespace std;
//�㷨����
typedef struct _tAlgParam
{
	int min_face;					//������С������ 40
	float scale_factor;				//������������ 0.709
	float cls_thre[3];				//��������Ԥ���������ֵ

}TAlgParam, *PTAlgParam;


typedef struct _tModelParam
{
	//ģ���ļ�
	vector<string> model_path;
	//��ͨ����ֵ
	vector<cv::Scalar> mean_value;
	//����ϵ��
	vector<float> scale_factor;
	//device
	torch::DeviceType device_type;
	int gpu_id;

	//�㷨����
	TAlgParam alg_param;
}TModelParam, *PTModelParam;
class MTCNN
{
public:
	//��ʼ�������
	int InitDetector(PTModelParam pparm);
	//�������
	int DetectFace(cv::Mat& src, vector<cv::Rect>& outFaces);
	// pnet �ֲ����
	int RunPNetLayer(cv::Mat& src, int scale_idx, std::vector<float>& outBoxes, std::vector<float>& outScores);

	int PrePNET(cv::Mat& src, float scalor, std::vector<torch::jit::IValue>& ti);

	int PostPNET(torch::jit::IValue& to, float scalor, std::vector<float>& outBoxes, std::vector<float>& outScores);

protected:
	int PNET(cv::Mat& src, std::vector<cv::Rect>& outFaces);
	int RNET(cv::Mat& src, std::vector<cv::Rect>& outFaces);
	int ONET(cv::Mat& src, std::vector<cv::Rect>& outFaces);

	//��������
	int GetPyrScalors(int W, int H, float scale_factor, int min_face_size);


	//pnetӳ��bbox
	int GenerateBBox(std::vector<float>& bbox, std::vector<int>& pos, float scalor) const;

	//���λ�
	int SquareBBox(std::vector<cv::Rect>& rect);
	//pad
	int PadBBox(cv::Rect rect, cv::Rect& rdst, cv::Rect& rsrc, int W, int H);

	//re-project to original image
	int ReProjectBBox(std::vector<float>& origin, std::vector<float>& bbox);

	//rnet onet�Ĺ���
	int RunRONet(torch::jit::script::Module net, cv::Mat& src, std::vector<cv::Rect>& outFaces, int IMAGE_SIZE);


private:
	vector<float> mvScales;
	TModelParam mtParam;
	//pnet
	torch::jit::script::Module mpPNET;
	//rnet
	torch::jit::script::Module mpRNET;
	//onet
	torch::jit::script::Module mpONET;
	//netoutput
	static const vector<string> pnet_output;
	static const vector<string> rnet_output;
	static const vector<string> onet_output;
	int NMS(std::vector<float>& bbox, std::vector<float>& score, float thre, int type) const;
};
