#include "YOLO_Segment.h"

void YOLO_Segment::ColorMerge(cv::Mat& frame, float& alpha, cv::Mat& mask, cv::Scalar color)
{
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (mask.at<float>(i, j) != 0)
			{
				cv::Vec3b values = frame.at<cv::Vec3b>(i, j);
				frame.at<cv::Vec3b>(i, j)[0] = alpha * values[0] + (1 - alpha) * color[0];
				frame.at<cv::Vec3b>(i, j)[1] = alpha * values[1] + (1 - alpha) * color[1];
				frame.at<cv::Vec3b>(i, j)[2] = alpha * values[2] + (1 - alpha) * color[2];
			}
		}
	}
}

void YOLO_Segment::MaskProcess(const int& iFrameWidth, const int& iFrameHeight, const cv::Mat& mask_in, cv::Rect &detection_box, cv::Mat& mask_out)
{
	int upsample_size = std::max(iFrameHeight, iFrameWidth);

	float scale_factor_x = (float)upsample_size / (float)mask_in.cols;
	float scale_factor_y = (float)upsample_size / (float)mask_in.rows;

	int ow = static_cast<int>(detection_box.width / scale_factor_x);
	int oh = static_cast<int>(detection_box.height / scale_factor_y);
	int cy = static_cast<int>(detection_box.y / scale_factor_y);
	int cx = static_cast<int>(detection_box.x / scale_factor_x);

	cv::Mat sigmod_mask = cv::Mat::zeros(mask_in.rows, mask_in.cols, CV_32F);
	for (int i = 0; i < mask_in.rows; i++)
	{
		for (int j = 0; j < mask_in.cols; j++)
		{
			if (i<cy || i>cy+oh || j<cx || j>cx+ow)
			{
				sigmod_mask.at<float>(i, j) = 0.0f;
			}
			else
			{
				float x = mask_in.at<float>(i, j);
				sigmod_mask.at<float>(i, j) = Sigmoid(x);
			}
		}
	}
	
	cv::Mat mask_resize;
	cv::resize(sigmod_mask, mask_resize, cv::Size(upsample_size, upsample_size), 0, 0, cv::INTER_LINEAR);
	cv::Mat thre_mask;
	cv::threshold(mask_resize, thre_mask, 0.5, 1, cv::THRESH_BINARY);

	if (iFrameHeight < iFrameWidth)
		mask_out = thre_mask.rowRange(0, iFrameHeight);
	else
		mask_out = thre_mask.colRange(0, iFrameWidth);
}

void YOLO_Segment::Preprocess(cv::Mat frame_in, cv::Mat& frame_resample)
{
	int w = frame_in.cols;
	int h = frame_in.rows;
	int _max = std::max(h, w);
	frame_resample = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame_in.copyTo(frame_resample(roi));

	fScaleFactorX = frame_resample.cols / static_cast<float>(input_dims[0][3]);
	fScaleFactorY = frame_resample.rows / static_cast<float>(input_dims[0][2]);
}

void YOLO_Segment::Postprocess(std::vector<Ort::Value>& ort_outputs)
{
	const float* fBoxdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_dims[0][1], output_dims[0][2], CV_32F, (float*)fBoxdata);
	cv::Mat det_output = dout.t(); 

	const float* fMaskdata = ort_outputs[1].GetTensorData<float>();
	cv::Mat mask_origin(32, 160 * 160, CV_32F, (float*)fMaskdata);
	cv::Mat mask_scores = det_output.colRange(84, 116);

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<int> mask_index;

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
			mask_index.push_back(i);
		}
	}

	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	masks.resize(indexes.size());

	for (size_t i = 0; i < indexes.size(); i++)
	{
		int index = indexes[i];
		cv::Mat mask_line = mask_scores.row(mask_index[index]) * mask_origin;
		cv::Mat mask_temp = mask_line.reshape(1, 160);
		MaskProcess(frame_out.cols, frame_out.rows, mask_temp, boxes[index], masks[i]);
	}

	label_colors.resize(indexes.size());
	for (int i = 0; i < indexes.size(); i++)
	{
		label_colors[i] = GetColor(classIds[indexes[i]]);
	}

	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];

		float alpha = 0.5;
		ColorMerge(frame_out, alpha, masks[i], label_colors[i]);

		cv::rectangle(frame_out, boxes[index], label_colors[i], 2, 8);
		cv::rectangle(frame_out, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), label_colors[i], -1);
		//std::string labels_confidence = labels[idx] + " " + std::to_string(confidences[index]);
		std::string labels_confidence = labels[idx];
		putText(frame_out, labels_confidence, cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 2, 8);
	}

}

std::vector<Ort::Value> YOLO_Segment::Run_model(cv::Mat& frame_resample)
{
	int input_pixels_num = input_dims[0][3] * input_dims[0][2] * 3;
	std::array<int64_t, 4> input_shape_info = { 1,3,input_dims[0][2],input_dims[0][3] };
	cv::Mat blob = cv::dnn::blobFromImage(frame_resample, 1 / 255.0, cv::Size(input_dims[0][3], input_dims[0][2]), cv::Scalar(0, 0, 0), true, false);
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value ort_input = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), input_pixels_num, input_shape_info.data(), input_shape_info.size());
	std::vector<Ort::Value> ort_outputs;
	const std::array<const char*, 1> inputNames{ input_node_names[0] };
	const std::array<const char*, 2> outputNames{ output_node_names[0],output_node_names[1] };

	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

	ort_outputs = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &ort_input, 1, outputNames.data(), outputNames.size());
	return ort_outputs;
}

cv::Mat YOLO_Segment::segment(cv::Mat frame_in)
{
	frame_in.copyTo(frame_out);

	cv::Mat frame_resample;

	Preprocess(frame_in, frame_resample);

	std::vector<Ort::Value> ort_outputs;
	ort_outputs = Run_model(frame_resample);

	Postprocess(ort_outputs);

	return frame_out;
}