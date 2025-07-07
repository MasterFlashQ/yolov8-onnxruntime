#include <iostream>
#include <queue>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>

#include "YOLO_Detector.h"
#include "YOLO_Segment.h"

std::queue<cv::Mat> frames;
std::mutex lock;
std::vector<std::string> labels{
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant",
		"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
		"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
		"kite",
		"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
		"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
		"donut",
		"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
		"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
		"scissors",
		"teddy bear", "hair drier", "toothbrush"
};

void CaptureFrame()
{
	cv::VideoCapture cap(0);

	assert(!cap.isOpened);

	while (true)
	{
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			continue;

		std::lock_guard<std::mutex> lock_guard(lock);
		frames.push(frame);
	}
	cap.release();

}

void PostProcess()
{
	std::string onnxpath_detect = "E:/YOLO/yolov8n.onnx";
	std::string onnxpath_segment = "E:/YOLO/yolov8n-seg.onnx";

	//YOLO_Detector* detector = new YOLO_Detector(onnxpath_detect, labels);

	//cv::namedWindow("yolo detect", cv::WINDOW_AUTOSIZE);

	YOLO_Segment* segmentor = new YOLO_Segment(onnxpath_segment,labels);

	cv::namedWindow("yolo segment", cv::WINDOW_AUTOSIZE);

	while (true)
	{
		cv::Mat frame, frame_processed;
		std::lock_guard<std::mutex> lockGuard(lock);
		if (frames.empty())
			continue;
		frame = frames.front();
		frames.pop();
		if (frame.empty())
			continue;

		//frame_processed = detector->detect(frame);
		//cv::imshow("yolo detect", frame_processed);

		frame_processed = segmentor->segment(frame);
		cv::imshow("yolo segment", frame_processed);
		cv::waitKey(1);
	}

	cv::destroyAllWindows();
	//delete detector;
	delete segmentor;
}

int main()
{
	std::thread(CaptureFrame).detach();
	std::thread(PostProcess).detach();
	while (true) {}
}
