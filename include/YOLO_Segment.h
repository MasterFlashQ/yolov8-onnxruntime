#pragma once
#ifndef YOLO_SEGMENT_H_
#define YOLO_SEGMENT_H_

#include "YOLO.h"

class YOLO_Segment :public YOLO
{
public:
	YOLO_Segment(std::string model_path,std::vector<std::string> labels) :YOLO(model_path,labels) {}
	cv::Mat segment(cv::Mat frame_in);
private:
	void Preprocess(cv::Mat frame_in, cv::Mat& frame_resample) override;
	void Postprocess(std::vector<Ort::Value>& ort_outputs) override;

	std::vector<Ort::Value> Run_model(cv::Mat& frame_resample) override;

	std::vector<cv::Mat> masks;

	void ColorMerge(cv::Mat& frame, float& alpha, cv::Mat& mask, cv::Scalar color);

	float Sigmoid(float& x)
	{
		return 1 / (1 + exp(-x));
	}
	void MaskProcess(const int& iFrameWidth, const int& iFrameHeight, const cv::Mat& mask_in, cv::Rect& detection_box, cv::Mat& mask_out);
};

#endif // !YOLO_SEGMENT_H_
