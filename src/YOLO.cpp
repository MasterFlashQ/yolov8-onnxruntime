#include "YOLO.h"

YOLO::YOLO(std::string model_path, std::vector<std::string> labels)
{
	this->labels = labels;

	std::wstring modelPath = std::wstring(model_path.begin(), model_path.end());
	env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

	session = Ort::Session(env, modelPath.c_str(), session_options);

	numInputNodes = session.GetInputCount();
	numOutputNodes = session.GetOutputCount();

	input_dims.resize(numInputNodes);
	output_dims.resize(numOutputNodes);
	input_node_names.resize(numInputNodes);
	output_node_names.resize(numOutputNodes);

	for (int i = 0; i < numInputNodes; i++)
	{
		input_node_names[i]=session.GetInputName(i, allocator);
		auto input_type_info = session.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		input_dims[i] = input_tensor_info.GetShape();
	}

	for (int i = 0; i < numOutputNodes; i++)
	{
		output_node_names[i] = session.GetOutputName(i, allocator);
		auto output_type_info = session.GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		output_dims[i] = output_tensor_info.GetShape();
	}

}

cv::Scalar YOLO::GetColor(int i)
{
	i = ((float)i / (float)labels.size()) * 255;
	//BGR
	cv::Scalar p;
	if (i < 32)
	{
		p[0] = 128 + 4 * i;
		p[1] = 0;
		p[2] = 0;
	}
	else if (i == 32)
	{
		p[0] = 255;
		p[1] = 0;
		p[2] = 0;
	}
	else if (i < 96)
	{
		p[0] = 255;
		p[1] = 4 + 4 * (i - 33);
		p[2] = 0;
	}
	else if (i == 96)
	{
		p[0] = 254;
		p[1] = 255;
		p[2] = 2;
	}
	else if (i < 159)
	{
		p[0] = 250 - 4 * (i - 97);
		p[1] = 255;
		p[2] = 6 + 4 * (i - 97);
	}
	else if (i == 159)
	{
		p[0] = 1;
		p[1] = 255;
		p[2] = 254;
	}
	else if (i < 224)
	{
		p[0] = 0;
		p[1] = 252 - ((i - 160) * 4);
		p[2] = 255;
	}
	else
	{
		p[0] = 0;
		p[1] = 0;
		p[2] = 252 - ((i - 224) * 4);
	}

	return p;
}