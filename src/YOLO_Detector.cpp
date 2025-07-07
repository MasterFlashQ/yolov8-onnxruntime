#include "YOLO_Detector.h"

void YOLO_Detector::Preprocess(cv::Mat frame_in, cv::Mat& frame_resample)
{
	int iFrameCols = frame_in.cols;
	int iFrameRows = frame_in.rows;
	int _max = std::max(iFrameRows, iFrameCols);
	frame_resample = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, iFrameCols, iFrameRows);
	frame_in.copyTo(frame_resample(roi));

	fScaleFactorX = frame_resample.cols / static_cast<float>(input_dims[0][3]);
	fScaleFactorY = frame_resample.rows / static_cast<float>(input_dims[0][2]);
}

void YOLO_Detector::Postprocess(std::vector<Ort::Value>& ort_outputs)
{
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_dims[0][1], output_dims[0][2], CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); 

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
		cv::Point classIdPoint;
		double score;
		cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		if (score > 0.25)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * fScaleFactorX);
			int y = static_cast<int>((cy - 0.5 * oh) * fScaleFactorY);
			int width = static_cast<int>(ow * fScaleFactorX);
			int height = static_cast<int>(oh * fScaleFactorY);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

	label_colors.resize(indexes.size());
	for (int i = 0; i < indexes.size(); i++)
	{
		label_colors[i] = GetColor(classIds[indexes[i]]);
	}

	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::rectangle(frame_out, boxes[index], label_colors[i], 2, 8);
		cv::rectangle(frame_out, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), label_colors[i], -1);
		//std::string labels_confidence = labels[idx] + " " + std::to_string(confidences[index]);
		std::string labels_confidence = labels[idx];
		putText(frame_out, labels_confidence, cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 2, 8);
	}
}

std::vector<Ort::Value> YOLO_Detector::Run_model(cv::Mat& frame_resample)
{
	int input_pixels_num = input_dims[0][3] * input_dims[0][2] * 3;
	std::array<int64_t, 4> input_shape_info = { 1,3,input_dims[0][2],input_dims[0][3] };
	cv::Mat blob = cv::dnn::blobFromImage(frame_resample, 1 / 255.0, cv::Size(input_dims[0][3], input_dims[0][2]), cv::Scalar(0, 0, 0), true, false);
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value ort_input = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), input_pixels_num, input_shape_info.data(), input_shape_info.size());
	std::vector<Ort::Value> ort_outputs;
	const std::array<const char*, 1> inputNames{ input_node_names[0] };
	const std::array<const char*, 1> outputNames{ output_node_names[0] };

	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

	ort_outputs = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &ort_input, 1, outputNames.data(), outputNames.size());
	return ort_outputs;
}

cv::Mat YOLO_Detector::detect(cv::Mat frame_in)
{
	frame_in.copyTo(frame_out);

	cv::Mat frame_resample;

	Preprocess(frame_in, frame_resample);

	std::vector<Ort::Value> ort_outputs;
	ort_outputs = Run_model(frame_resample);

	Postprocess(ort_outputs);

	return frame_out;
}