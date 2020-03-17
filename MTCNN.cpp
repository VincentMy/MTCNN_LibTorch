//
// Created by beichen2012 on 18-12-21.
//

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

//#define USE_PNET_MULTI_THREAD
#ifndef PNET_THREAD_NUM
#define PNET_THREAD_NUM 2
#endif

int MTCNN::InitDetector(PTModelParam pparm)
{
	mtParam = *pparm;
	try {
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
	pparm->device_type = at::kCUDA;
	torch::Device device(pparm->device_type, 0);
	mpPNET.to(device);
	mpRNET.to(device);
	mpONET.to(device);
	return 0;
}

//检测人脸，其中cv::Rect表示的是矩阵
int MTCNN::DetectFace(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
	//int64_t tp, tr, to;
	outFaces.clear();
	int ret = 0;
	ret = PNET(src, outFaces);
	LOGI("Pnet time cost");
	if (ret < 0)
		return ret;

	ret = RNET(src, outFaces);
	LOGI("Rnet time cost");

	if (ret < 0)
		return ret;

	ret = ONET(src, outFaces);
	LOGI("Onet time cost");

	return ret;
}


int MTCNN::PNET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
	//1. 计算金字塔
	//BTimer t;
	//获取图片缩放比率 mvScales
	GetPyrScalors(src.cols, src.rows, mtParam.alg_param.scale_factor, mtParam.alg_param.min_face);
	//LOGI("pyramid time cost: {}", t.elapsed());
	//LOGI("pyramid total num: {}", mvScales.size());
	if (mvScales.size() <= 0)
		return -1;

	//3. 分层计算
	//表示每次缩放后的box和score进行存放
	std::vector<std::vector<float>> layerBoxes;
	std::vector<std::vector<float>> layerScores;
	//改变容器的大小
	layerBoxes.resize(mvScales.size());
	layerScores.resize(mvScales.size());
	//根据缩放的次数，执行多次RunPNetLayer
	for (int i = 0; i < mvScales.size(); i++)
	{
		RunPNetLayer(src, i, layerBoxes[i], layerScores[i]);
	}
	//5. NMS
	std::vector<float> bboxes;
	std::vector<float> score;
	//把pnet处理后的结果填充到bboxes和score中
	for (int i = 0; i < layerBoxes.size(); i++)
	{
		bboxes.insert(bboxes.end(), layerBoxes[i].begin(), layerBoxes[i].end());
		score.insert(score.end(), layerScores[i].begin(), layerScores[i].end());
	}

	//t.reset();
	//再次进行nms
	NMS(bboxes, score, 0.7, 0);
	//LOGI("Pnet NMS time cost: {}", t.elapsed());


	// 组合结果
	for (int i = 0; i < bboxes.size() / 4; i++)
	{
		cv::Rect r = cv::Rect(bboxes[4 * i], bboxes[4 * i + 1], bboxes[4 * i + 2], bboxes[4 * i + 3]);
		outFaces.push_back(r);
	}

	return 0;
}
/*
	此函数主要是对图片矩阵进行适当的预处理，包括：
	1、图片矩阵缩放。
	2、图片矩阵类型变换
	3、均值化，减去均值
	4、通道变换 BGR->RGB，并增加一个batch列
	5、处理后的图片存入ti中
*/
int MTCNN::PrePNET(cv::Mat& src, float scalor, std::vector<torch::jit::IValue>& ti)
{
	//1. 这一层的Image
	//BTimer t;
	// 获取当前缩放后的长宽
	int nw = src.cols * scalor;
	int nh = src.rows * scalor;
	cv::Mat image;
	//t.reset();
	cv::resize(src, image, { nw, nh });
	//LOGI("pnet layer resize : {} us", t.elapsed_micro());
	//定义一个空的tensor,方便后面赋值
	torch::Tensor input = torch::zeros({ image.rows, image.cols, image.channels() });

	cv::Mat img_float;
	image.convertTo(img_float, CV_32F);
	//图像减去均值，此处的均值为127.5,后续可以使用opencv计算均值
	img_float -= mtParam.mean_value[0];
	if (std::abs(1.0f - mtParam.scale_factor[0]) > 0.001)
	{
		img_float *= mtParam.scale_factor[0];
	}
	//numel()表示返回数组中元素的个数
	auto nn = input.numel();
	//memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中
	memcpy(input.data_ptr(), img_float.data, input.numel() * sizeof(float));
	//input = input.to(torch::Device(mtParam.device_type, mtParam.gpu_id));
	input = input.to(at::kCUDA);
	//改变通道由BGR变为RGB
	input = input.permute({ 2, 0, 1 });
	//添加一个batch位
	input.unsqueeze_(0);
	//清空ti容器
	ti.clear();
	ti.emplace_back(std::move(input));
	return 0;
}

int MTCNN::PostPNET(torch::jit::IValue& to, float scalor, std::vector<float>& outBoxes, std::vector<float>& outScores)
{
	auto ot = to.toTuple();

	std::vector<torch::jit::IValue> inputs = ot->elements();
	//input的size是2
	std::cout << "pnet inputs size :{%d}" << inputs.size() << std::endl;//2
	auto cls = inputs[0].toTensor().cpu();
	auto reg = inputs[1].toTensor().cpu();
	std::cout << "pnet cls size :{%d}" << cls.sizes() << std::endl;//[1, 2, 157, 211]
	std::cout << "pnet reg size :{%d}" << reg.sizes() << std::endl;//[1, 4, 157, 211]
	//size(n) 表示第n维的大小
	auto cls_map_h = cls.size(2);//表示h的size
	auto cls_map_w = cls.size(3);
	//accessor<dtype,dim>,访问Tensor中的数据，并将数据返回为dtype类型，此处就是把cls和reg中的数据转化为float类型
	auto cls_viewer = cls.accessor<float, 4>();
	auto reg_viewer = reg.accessor<float, 4>();

	std::vector<float> bboxes;	//M * 4
	std::vector<float> score;	//M * 1
	std::vector<int> pos;		//M * 2 position，表示图片上的具体位置
	for (int y = 0; y < cls_map_h; y++)
	{
		for (int x = 0; x < cls_map_w; x++)
		{
			if (cls_viewer[0][1][y][x] > mtParam.alg_param.cls_thre[0]) //正样本的比例大于阈值
			{
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

	//LOGI("pnet layer feature map extract time: {}", t.elapsed());
	if (bboxes.size() <= 0)
		return 0;

	//5. 坐标映射回原图
	//t.reset();
	GenerateBBox(bboxes, pos, scalor);
	//LOGI("pnet GenerateBBox time: {}", t.elapsed());

	//6. NMS
	//t.reset();
	NMS(bboxes, score, 0.5, 0);
	//LOGI("pnet NMS time: {}", t.elapsed());

	//7. 填充输出
	outBoxes = std::move(bboxes);
	outScores = std::move(score);
	return 0;
}

int MTCNN::RunPNetLayer(cv::Mat& src, int scale_idx, std::vector<float>& outBoxes, std::vector<float>& outScores)
{
	//获取第scale_idx次缩放的比率
	float scalor = mvScales[scale_idx];
	std::vector<torch::jit::IValue> inputs;
	PrePNET(src, scalor, inputs);
	//获得输出
	auto&& to = mpPNET.forward(inputs);
	//此处的outBoxes和outScores是空容器
	//把预测后的bbox进行处理到原图，然后进行nms，最后存入outBoxes和outScores中
	PostPNET(to, scalor, outBoxes, outScores);

	return 0;
}
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

	//升序
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
		//最后的score是最大的
		TBBox tb = tbox.back();
		order = tb.order;
		tbox.pop_back();
		if (order < 0)
			continue;

		//保存当前box的索引
		heros.push_back(order);
		//遍历时忽略掉
		tb.exist = false;

		for (auto &i : tbox)
		{
			if (!i.exist)
				continue;
			//计算当前box与tb的交并比
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

	//结果
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


//生成bbox
int MTCNN::GenerateBBox(std::vector<float>& bbox, std::vector<int>& pos, float scalor) const
{
	//0.
	static const int stride = 2;
	static const int cellsize = PNET_IMAGE_SIZE;
	//1. 转换成cv::Mat(后续可考虑使用eigen)
	int w = 4;
	//表示分成4行
	int h = bbox.size() / w;			//h = pos.size() / 2;
	float gx;
	float gy;
	float* ptr;
	//位置信息的数量一定要和bbox的数量一致
	if (h != pos.size() / 2)
	{
		bbox.clear();
		return -1;
	}
	//h*w的矩阵，标识偏移量
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > reg(bbox.data(), h, w);
	//h*2的矩阵，标识Patch with height
	Map<Matrix<int, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > gp(pos.data(), h, 2);
	MatrixXf Gstart = gp.cast<float>();
	Gstart.array() *= (stride / scalor);


	//reg 缩放回原图尺寸
	reg.col(0) *= (cellsize / scalor);
	reg.col(1) *= (cellsize / scalor);
	reg.col(2).array() = reg.col(2).array().exp();
	reg.col(3).array() = reg.col(3).array().exp();
	reg.col(2).array() *= (cellsize / scalor);
	reg.col(3).array() *= (cellsize / scalor);

	//reg加上起点及终点偏移量
	reg.col(0) += Gstart.col(0);
	reg.col(1) += Gstart.col(1);
	//
	return 0;
}
//转换框为正方形
//通过计算正方形左上角的点，然后边长为为之前长宽的最大值
int MTCNN::SquareBBox(std::vector<cv::Rect>& rect)
{
	int size;
	cv::Rect r;
	for (auto& i : rect)
	{

		size = std::max(i.width, i.height);

		r.x = i.x + i.width * 0.5 - size * 0.5;
		r.y = i.y + i.height * 0.5 - size * 0.5;
		r.width = size;
		r.height = size;
		i = r;
	}
	return 0;
}
/*
对pnet返回的bbox进行截取，对于大于原图的x,y标记为w和h,对于小于原图像坐标的x,y标记为0
rsrc：表示bbox截取后的左上角和右小角坐标
rdst：表示bbox小于图像边界的距离(前两项dx0,dy0)，和实际bbox的长宽(后两项dx1和dy1)
*/
int MTCNN::PadBBox(cv::Rect rect, cv::Rect & rdst, cv::Rect & rsrc, int W, int H)
{
	//original，其中rect是pnet的输出
	int x0 = rect.x;
	int y0 = rect.y;
	int x1 = rect.width + x0;
	int y1 = rect.height + y0;

	//src, dst sx/sy表示的是坐标。dx/dy表示是距离
	int sx0, sy0, sx1, sy1;
	int dx0, dy0, dx1, dy1;
	int minx, miny;
	sx0 = x0;
	dx0 = 0;
	sy0 = y0;
	dy0 = 0;

	sx1 = x1;
	dx1 = rect.width;
	sy1 = y1;
	dy1 = rect.height;

	//边界判断
	if (x0 < 0)
	{
		sx0 = 0;
		dx0 = -x0;//表示距离0点的宽度
	}
	if (y0 < 0)
	{
		sy0 = 0;
		dy0 = -y0; //表示边界框距离0点的长度
	}
	if (x1 > W - 1) //x大于图片的宽度
	{
		sx1 = W - 1; //bbox在图像右边界的坐标
		dx1 = sx1 - sx0; //bbox的宽度
	}
	if (y1 > H - 1) //y大于图片的长度
	{
		sy1 = H - 1; //bbox在图像下边界的坐标
		dy1 = sy1 - sy0; //bbox的长度
	}
	//src,dst的shape要一致
	//获取最小bbox长度
	miny = std::min(dy1 - dy0, sy1 - sy0);
	dy1 = dy0 + miny;//实际bbox的长度
	sy1 = sy0 + miny; //实际bbox下边界的y坐标
	minx = std::min(dx1 - dx0, sx1 - sx0);
	dx1 = dx0 + minx;
	sx1 = sx0 + minx;

	//填充输出，此处的输出是根据pnet的返回值来界定的
	rsrc = { sx0, sy0, sx1 - sx0, sy1 - sy0 };
	rdst = { dx0, dy0, dx1 - dx0, dy1 - dy0 };
	return 0;
}
int MTCNN::ReProjectBBox(std::vector<float>& origin, std::vector<float>& bbox)
{
	//wrap to matrix
	int N = bbox.size() / 4;

	//
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> mo(origin.data(), N, 4);
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> mb(bbox.data(), N, 4);

	//scale
	mb.col(0).array() *= mo.col(2).array();
	mb.col(1).array() *= mo.col(3).array();
	mb.col(0) += mo.col(0);
	mb.col(1) += mo.col(1);

	//w,h
	mb.col(2).array() = mb.col(2).array().exp();
	mb.col(3).array() = mb.col(3).array().exp();
	mb.col(2).array() *= mo.col(2).array();
	mb.col(3).array() *= mo.col(3).array();

	return 0;
}
/*
typedef struct CvRect
　　{
　　int x; //方形的左上角的x-坐标
　　int y; // 方形的左上角的y-坐标
　　int width; // 宽
　　int height; // 高
　　}
*/
int MTCNN::RunRONet(torch::jit::script::Module net,
	cv::Mat & src, //原图
	std::vector<cv::Rect>& outFaces, //pnet输出
	int IMAGE_SIZE)
{
	int stage = 1;
	if (IMAGE_SIZE == 48)
		stage = 2;
	//获得均值
	cv::Scalar mean = mtParam.mean_value[stage];
	//获取缩放比率
	float scalor = mtParam.scale_factor[stage];
	//获得阈值
	float cls_thre = mtParam.alg_param.cls_thre[stage];

	//0.其中outFaces表示pnet的输出
	if (outFaces.size() <= 0)
	{
		return 0;
	}
	int W = src.cols;
	int H = src.rows;
	//1. 生成bbox。把bbox框转换为正方形
	SquareBBox(outFaces);

	//2. pad
	cv::Rect rdst, rsrc;
	int size;

	std::vector<cv::Mat> imgs;
	std::vector<float> bboxOriginal;
	for (int i = 0; i < outFaces.size(); i++)
	{
		size = outFaces[i].width;
		cv::Mat crop = cv::Mat::zeros(cv::Size(size, size), CV_8UC3);
		PadBBox(outFaces[i], rdst, rsrc, W, H);
		if (rdst.x < 0 || rdst.y < 0 || rsrc.x < 0 || rsrc.y < 0 ||
			rdst.width > size || rdst.height > size ||
			rsrc.width > W || rsrc.height > H)
		{
			continue;
		}
		//copy
		//把坐标点都大于0，并且长宽都大于0的部分存入crop中
		src(rsrc).copyTo((cv::Mat)crop(rdst));
		//
		cv::resize(crop, crop, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
		cv::Mat crop_float;
		crop.convertTo(crop_float, CV_32F);
		crop_float -= mean;
		if (std::abs(1.0f - scalor) > 0.001)
			crop_float *= scalor;
		imgs.emplace_back(crop_float);

		bboxOriginal.push_back(outFaces[i].x);
		bboxOriginal.push_back(outFaces[i].y);
		bboxOriginal.push_back(outFaces[i].width);
		bboxOriginal.push_back(outFaces[i].height);
	}

	//3. predict
	int N = imgs.size();
	torch::Tensor input = torch::zeros({ N, IMAGE_SIZE, IMAGE_SIZE, 3 });
	//head表示内存位置
	float* head = input.data<float>();
	for (int i = 0; i < imgs.size(); i++)
	{
		memcpy(head, imgs[i].data, sizeof(float) * 3 * IMAGE_SIZE * IMAGE_SIZE);
		head += 3 * IMAGE_SIZE * IMAGE_SIZE;
	}
	//input = input.to(torch::Device(mtParam.device_type, mtParam.gpu_id));
	input = input.to(at::kCUDA);
	input = input.permute({ 0, 3, 1, 2 });


	std::vector<torch::jit::IValue> inputs = { input };
	

	auto&& out = net.forward(inputs);
	auto ot = out.toTuple();
	inputs = ot->elements();
	auto cls = inputs[0].toTensor().cpu();
	auto reg = inputs[1].toTensor().cpu();

	std::vector<float> bboxes;	//用于标识回归框
	std::vector<float> score;	//分数
	std::vector<int> keep_index;	//位置
	std::vector<float> resBBoxes;	// 原框及最终框（复用）

	auto cls_viewer = cls.accessor<float, 2>();//N * 2
	auto reg_viewer = reg.accessor<float, 2>();//N * 4

	for (int i = 0; i < N; i++)
	{
		if (cls_viewer[i][1] > cls_thre)
		{
			score.push_back(cls_viewer[i][1]);
			keep_index.push_back(i);

			bboxes.push_back(reg_viewer[i][0]);
			bboxes.push_back(reg_viewer[i][1]);
			bboxes.push_back(reg_viewer[i][2]);
			bboxes.push_back(reg_viewer[i][3]);
		}
	}

	outFaces.clear();
	if (score.size() <= 0)
		return 0;

	// reproject to original
	for (auto& i : keep_index)
	{
		resBBoxes.push_back(bboxOriginal[4 * i]);
		resBBoxes.push_back(bboxOriginal[4 * i + 1]);
		resBBoxes.push_back(bboxOriginal[4 * i + 2]);
		resBBoxes.push_back(bboxOriginal[4 * i + 3]);
	}
	bboxOriginal = std::move(resBBoxes);
	keep_index.clear();
	ReProjectBBox(bboxOriginal, bboxes);

	//NMS
	if (IMAGE_SIZE == 24)
		NMS(bboxes, score, 0.6, 0);
	else
		NMS(bboxes, score, 0.6, 1);

	//7. output
	outFaces.clear();
	for (int i = 0; i < bboxes.size() / 4; i++)
	{
		cv::Rect r = cv::Rect(bboxes[4 * i], bboxes[4 * i + 1], bboxes[4 * i + 2], bboxes[4 * i + 3]);
		outFaces.push_back(r);
	}
	return 0;
}

int MTCNN::RNET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
	return RunRONet(mpRNET, src, outFaces, RNET_IMAGE_SIZE);
}

int MTCNN::ONET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
	return RunRONet(mpONET, src, outFaces, ONET_IMAGE_SIZE);
	return 0;
}
/*
	此方法用来计算图片缩放的比率，并把所有比率值存入mvScales
*/
int MTCNN::GetPyrScalors(int W, int H, float scale_factor, int min_face_size)
{
	//0, 准备金字塔层数
	mvScales.clear();
	//获取长和宽的最小值，用于后面进行缩放
	float minl = H > W ? W : H;
	
	float base_scale = float(PNET_IMAGE_SIZE) / float(min_face_size);
	int face_count = 0;
	while (true)
	{
		//计算缩放比率，其中pow(a,b)表示a的b次方
		float s = base_scale * (float)pow(double(scale_factor), double(face_count));
		//如果缩放后的图片尺寸小于pnet检测的最小人脸，退出
		if (floor(double(minl * s)) < PNET_IMAGE_SIZE)
			break;
		mvScales.push_back(s);
		face_count++;
	}
	return 0;
}
