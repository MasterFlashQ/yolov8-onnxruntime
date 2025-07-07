#pragma once
#ifndef YOLO_DETECTOR_H_
#define YOLO_DETECTOR_H_

#include "YOLO.h"

class YOLO_Detector :public YOLO
{
public:
	YOLO_Detector(std::string model_path, std::vector<std::string> labels):YOLO(model_path,labels) {}
	cv::Mat detect(cv::Mat frame_in);
private:
	void Preprocess(cv::Mat frame_in, cv::Mat& frame_resample) override;
	void Postprocess(std::vector<Ort::Value>& ort_outputs) override;

	std::vector<Ort::Value> Run_model(cv::Mat& frame_resample) override;
};

#endif // !YOLO_DETECTOR_H_
