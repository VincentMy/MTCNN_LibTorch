#include "MTCNN.h"
#include <numeric>
#include <utility>

#include <spdhelper.hpp>
//#include <BTimer.hpp>
//#include <future>
#include <Eigen/Eigen>
using namespace Eigen;
//#include "torchutils.h"

#define PNET_IMAGE_SIZE 12
#define RNET_IMAGE_SIZE 24
#define ONET_IMAGE_SIZE 48

int MTCNN::InitDetector(PTModelParam pparm)
{
	mtParam = *pparm;
	try{
	//pnet
	mpPNET = torch::jit::load(pparm->model_path[0]);
	//rnet
	mpRNET = torch::jit::load(pparm->model_path[1]);
	//onet
	mpONET = torch::jit::load(pparm->model_path[2]);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	//device
	pparm->device_type = torch::kCPU;
	torch::Device device(pparm->device_type, 0);
	mpPNET.to(device);
	mpRNET.to(device);
	mpONET.to(device);
	return 0;
}
int MTCNN::DetectFace(cv::Mat& src, vector<cv::Rect>& outFaces)
{
	outFaces.clear();
	int ret = 0;
	ret = PNET(src, outFaces);
	LOGI("ret:{}", ret);
	return 0;
}

int MTCNN::PNET(cv::Mat& src, vector<cv::Rect>& outFaces) 
{
	//���������
	GetPyrScalors(src.cols,src.rows,mtParam.alg_param.scale_factor,mtParam.alg_param.min_face);
	if (mvScales.size() <= 0)
		return -1;
	//�ֲ����
	std::vector<std::vector<float>> layerBoxes;
	std::vector<std::vector<float>> layerScores;
	//�������ŵĴ���������vector�Ĵ�С
	layerBoxes.resize(mvScales.size());
	layerScores.resize(mvScales.size());
	
	for (int i = 0; i < mvScales.size(); i++) {
		RunPNetLayer(src, i, layerBoxes[i], layerScores[i]);
	}
	//5. NMS
	std::vector<float> bboxes;
	std::vector<float> score;
	for (int i = 0; i < layerBoxes.size(); i++)
	{
		bboxes.insert(bboxes.end(), layerBoxes[i].begin(), layerBoxes[i].end());
		score.insert(score.end(), layerScores[i].begin(), layerScores[i].end());
	}
	LOGI("PNET_4");
	//t.reset();
	NMS(bboxes, score, 0.7, 0);
	//LOGI("Pnet NMS time cost: {}", t.elapsed());


	// ��Ͻ��
	for (int i = 0; i < bboxes.size() / 4; i++)
	{
		cv::Rect r = cv::Rect(bboxes[4 * i], bboxes[4 * i + 1], bboxes[4 * i + 2], bboxes[4 * i + 3]);
		outFaces.push_back(r);
	}
	
	return 0;
}

//�˷�����Ҫ�Ƕ�����ͼƬ����Ԥ����1�����š�2������ֵ(127.5)��3�����batch�С�4������ti
int MTCNN::PrePNET(cv::Mat& src, float scalor, std::vector<torch::jit::IValue>& ti)
{
	//�Ե�ǰ��image��������
	int nw = src.cols * scalor;
	int nh = src.rows * scalor;
	cv::Mat image;
	//��ͼƬ����resize��ָ����С
	cv::resize(src, image, { nw, nh });
	torch::Tensor input = torch::zeros({ image.rows,image.cols,image.channels() });
	cv::Mat img_float;
	//opencv�о�����������ת��
	image.convertTo(img_float, CV_32F);
	//ͼƬ��ȥ127.5
	img_float -= mtParam.mean_value[0];
	if (std::abs(1.0f - mtParam.scale_factor[0]) > 0.001) {
		img_float *= mtParam.scale_factor[0];
	}
	//numel()��ʾ����������Ԫ�صĸ���
	auto nn = input.numel();
	//memcpyָ����c��c++ʹ�õ��ڴ濽��������memcpy�����Ĺ����Ǵ�Դsrc��ָ���ڴ��ַ����ʼλ�ÿ�ʼ����n���ֽڵ�Ŀ��dest��ָ���ڴ��ַ����ʼλ����
	memcpy(input.data_ptr(),img_float.data, input.numel() * sizeof(float));
	LOGI("device type{}",mtParam.device_type);
	//input = input.to(torch::Device(mtParam.device_type, mtParam.gpu_id));
	//input = input.to(torch::kCPU);
	//����input��ͨ����bgr->rgb
	input = input.permute({ 2,0,1 });
	//���һ�У���0��
	input.unsqueeze_(0);
	ti.clear();
	//�൱�ڰ�input�е����ݼ��е�ti��
	ti.emplace_back(std::move(input));
	return 0;

}

int MTCNN::PostPNET(torch::jit::IValue& to, float scalor, std::vector<float>& outBoxes, std::vector<float>& outScores) 
{
	auto ot = to.toTuple();
	std::vector<torch::jit::IValue> inputs = ot->elements();
	//����
	auto cls = inputs[0].toTensor().cpu();
	//�ع�
	auto reg = inputs[1].toTensor().cpu();

	auto cls_map_h = cls.size(2);
	auto cls_map_w = cls.size(3);
	//accessor<dtype,dim>,����Tensor�е����ݣ��������ݷ���Ϊdtype����
	auto cls_viewer = cls.accessor<float, 4>();
	auto reg_viewer = reg.accessor<float, 4>();

	std::vector<float> bboxes; //M * 4
	std::vector<float> score; //M * 1
	std::vector<int> pos;	 //M * 2

	for (int y = 0; y < cls_map_h; y++) {
		for (int x = 0; x < cls_map_w; x++) {
			if (cls_viewer[0][1][y][x] > mtParam.alg_param.cls_thre[0]) {
				score.push_back(cls_viewer[0][1][y][x]);

				pos.push_back(x);
				pos.push_back(y);

				bboxes.push_back(reg_viewer[0][0][y][x]);
				bboxes.push_back(reg_viewer[0][1][y][x]);
				bboxes.push_back(reg_viewer[0][2][y][x]);
				bboxes.push_back(reg_viewer[0][3][y][x]);
			}
		}
	}
	if (bboxes.size() <= 0)
		return 0;
	//����ӳ���ԭ����
	GenerateBBox(bboxes,pos,scalor);
	NMS(bboxes, score, 0.5, 0);
	//������
	outBoxes = std::move(bboxes);
	outScores = std::move(score);
	return 0;
}

//nms
int MTCNN::NMS(std::vector<float>& bbox, std::vector<float>& score, float thre, int type) const
{
	typedef struct TBBox {
		int x1;
		int y1;
		int x2;
		int y2;
		int order;
		float score;
		float area;
		bool exist;
	}TBBox;
	std::vector<TBBox> tbox;
	for (int i = 0; i < score.size(); i++)
	{
		TBBox b;
		b.x1 = bbox[i * 4 + 0];
		b.y1 = bbox[i * 4 + 1];
		b.x2 = bbox[i * 4 + 2] + b.x1;
		b.y2 = bbox[i * 4 + 3] + b.y1;

		b.order = i;
		b.score = score[i];
		b.exist = true;
		b.area = bbox[i * 4 + 2] * bbox[i * 4 + 3];
		tbox.push_back(b);
	}

	//����
	std::sort(tbox.begin(), tbox.end(), [](const TBBox& left, const TBBox& right) {
		return left.score < right.score;
	});

	std::vector<int> heros;
	int order = 0;
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;

	while (tbox.size() > 0)
	{
		//����score������
		TBBox tb = tbox.back();
		order = tb.order;
		tbox.pop_back();
		if (order < 0)
			continue;

		//���浱ǰbox������
		heros.push_back(order);
		//����ʱ���Ե�
		tb.exist = false;

		for (auto &i : tbox)
		{
			if (!i.exist)
				continue;
			//���㵱ǰbox��tb�Ľ�����
			auto& ti = i;
			maxX = ti.x1 > tb.x1 ? ti.x1 : tb.x1;
			maxY = ti.y1 > tb.y1 ? ti.y1 : tb.y1;
			minX = ti.x2 < tb.x2 ? ti.x2 : tb.x2;
			minY = ti.y2 < tb.y2 ? ti.y2 : tb.y2;

			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (type == 0)
			{
				//IOU
				IOU = IOU / (tb.area + ti.area - IOU);
			}
			else
			{
				//IOM
				IOU = IOU / (tb.area > ti.area ? ti.area : tb.area);
			}
			//compare
			if (IOU > thre)
			{
				ti.exist = false;
				ti.order = -1;
			}
		}
	}

	//���
	std::vector<float> outbox;
	std::vector<float> outscore;
	for (auto& i : heros)
	{
		outscore.push_back(score[i]);
		outbox.push_back(bbox[4 * i]);
		outbox.push_back(bbox[4 * i + 1]);
		outbox.push_back(bbox[4 * i + 2]);
		outbox.push_back(bbox[4 * i + 3]);
	}

	bbox.clear();
	bbox = std::move(outbox);
	score.clear();
	score = std::move(outscore);
	return 0;
}

//����bbox
int MTCNN::GenerateBBox(std::vector<float>& bbox, std::vector<int>& pos, float scalor) const
{
	//0.
	static const int stride = 2;
	static const int cellsize = PNET_IMAGE_SIZE;
	//1. ת����cv::Mat(�����ɿ���ʹ��eigen)
	int w = 4;
	int h = bbox.size() / w;			//h = pos.size() / 2;
	float gx;
	float gy;
	float* ptr;
	if (h != pos.size() / 2)
	{
		bbox.clear();
		return -1;
	}
	//h*w�ľ��󣬱�ʶƫ����
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > reg(bbox.data(), h, w);
	//h*2�ľ��󣬱�ʶPatch with height
	Map<Matrix<int, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > gp(pos.data(), h, 2);
	MatrixXf Gstart = gp.cast<float>();
	Gstart.array() *= (stride / scalor);


	//reg ���Ż�ԭͼ�ߴ�
	reg.col(0) *= (cellsize / scalor);
	reg.col(1) *= (cellsize / scalor);
	reg.col(2).array() = reg.col(2).array().exp();
	reg.col(3).array() = reg.col(3).array().exp();
	reg.col(2).array() *= (cellsize / scalor);
	reg.col(3).array() *= (cellsize / scalor);

	//reg������㼰�յ�ƫ����
	reg.col(0) += Gstart.col(0);
	reg.col(1) += Gstart.col(1);
	//
	return 0;
}


int MTCNN::RunPNetLayer(cv::Mat& src, int scale_idx, std::vector<float>& outBoxes, std::vector<float>& outScores)
{
	//��ȡĳһ����ϵ��
	float scalor = mvScales[scale_idx];
	std::vector<torch::jit::IValue> inputs;
	PrePNET(src, scalor, inputs);
	auto&& to = mpPNET.forward(inputs);
	//PostPNET(to, scalor, outBoxes, outScores);
	return 0;
}
//vector::clear()��������������������е����ݣ������ָ��Ļ�Ӧ�ò�����һ�ַ�ʽ
//�η�����������ȡͼ�������ϵ���ģ���mvScales�д���������ϵ��
int MTCNN::GetPyrScalors(int W, int H, float scale_factor, int min_face_size)
{
	//׼������������
	mvScales.clear();
	//���ظߺͿ�����С��
	float minl = H > W ? W : H;
	float base_scale = float(PNET_IMAGE_SIZE) / float(min_face_size);
	int face_count = 0;
	while (true)
	{
		//pow(a,b)��ʾ��a��b�η�,�Ӵ���Ҫ��������ϵ��
		float s = base_scale * (float)pow(double(scale_factor), double(face_count));
		//floor(a)��ʾ��a��������ȡ��
		if (floor(double(minl*s)) < PNET_IMAGE_SIZE)
			break;
		//��������ϵ����pus_back()��ʾ��ĩβ����Ԫ��
		mvScales.push_back(s);
		face_count++;
	}
	return 0;
}

