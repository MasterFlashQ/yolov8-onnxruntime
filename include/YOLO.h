#pragma once
#ifndef YOLO_H_
#define YOLO_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class YOLO
{
public:
	YOLO(std::string model_path, std::vector<std::string> labels);

	~YOLO()
	{
		session_options.release();
		session.release();
	}

protected:
	std::vector<std::string> labels;

	virtual std::vector<Ort::Value> Run_model(cv::Mat& frame_resample)=0;

	Ort::Session session{ nullptr };
	Ort::SessionOptions session_options;
	Ort::Env env;
	Ort::AllocatorWithDefaultOptions allocator;

	int numInputNodes, numOutputNodes;

	std::vector<std::vector<int64_t>> input_dims, output_dims;
	std::vector<char*> input_node_names, output_node_names;

	virtual void Preprocess(cv::Mat frame_in, cv::Mat& frame_resample)=0;
	virtual void Postprocess(std::vector<Ort::Value>& ort_outputs)=0;

	float fScaleFactorX, fScaleFactorY;
	cv::Mat frame_out;

	cv::Scalar GetColor(int i);
	std::vector<cv::Scalar> label_colors;
};

#endif // !YOLO_H_
