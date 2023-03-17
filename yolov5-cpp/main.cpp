#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include<locale.h>
#include<tchar.h>
#include<vector>
#include<conio.h>
#include <Windows.h>
//#include <opencv2/dnn.hpp>
//#include <opencv2/dnn/shape_utils.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <string>

//#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda)
{
    auto result = cv::dnn::readNet("yolov5s.onnx");
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        std::cout << "ok!!!!!!!!!!!!\n";
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

char* ConvertFromUnicode(const WCHAR* pszWchar)
{
	int		nLen;
	char* pszChar;

	//charに必要な文字数の取得
	nLen = ::WideCharToMultiByte(CP_THREAD_ACP, 0, pszWchar, -1, NULL, 0, NULL, NULL);
	pszChar = new char[nLen];
	if (pszChar == NULL)
		return	NULL;

	//変換
	nLen = ::WideCharToMultiByte(CP_THREAD_ACP, 0, pszWchar, wcslen(pszWchar) + 1, pszChar, nLen, NULL, NULL);
	if (nLen)
		return	pszChar;

	delete	pszChar;

	return	NULL;
}
/*
void Evaluation_image(std::vector<WCHAR> fileName)
{
	FILE* fp;
	errno_t csv_error;
	string sourceFileName;
	char FileName[256];
	char outputCsvName[256];

	int normal = 0;		//正常かどうか(1であれば正常)
	int abnormal = 0;	//異常かどうか(1であれば異常)
	double Pnormal = 0, Pabnormal = 0; //識別率



	String modelConfiguration = "yolov2-tiny_yamamoto.cfg";	//cfgファイル




	//! [Initialize network]
	dnn::Net net = readNetFromDarknet(modelConfiguration, (String)modelBinary);
	//! [Initialize network]

	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "cfg-file:     " << modelConfiguration << endl;
		cerr << "weights-file: " << modelBinary << endl;
		cerr << "Models can be downloaded here:" << endl;
		cerr << "https://pjreddie.com/darknet/yolo/" << endl;
		exit(-1);
	}

	//VideoCapture cap;
	//if (parser.get<String>("source").empty())
	//{
	//	int cameraDevice = parser.get<int>("camera_device");
	//	cap = VideoCapture(cameraDevice);
	//	if (!cap.isOpened())
	//	{
	//		cout << "Couldn't find camera: " << cameraDevice << endl;
	//		return -1;
	//	}
	//}
	//else
	//{
	//	cap.open(parser.get<String>("source"));
	//	if (!cap.isOpened())
	//	{
	//		cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
	//		return -1;
	//	}
	//}

	vector<string> classNamesVec;
	ifstream classNamesFile("evaluation.names");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	//for (;;)
	//{
	//	Mat frame;
	//	cap >> frame; // get a new frame from camera/video or read image

	//	if (frame.empty())
	//	{
	//		waitKey();
	//		break;
	//	}
	Mat source;
	sourceFileName = ConvertFromUnicode(&fileName[0]);
	sprintf_s(FileName, 256, "G:\\マイドライブ\\image_mae/%s", sourceFileName);
	source = imread(FileName);
	if (source.empty()) {
		printf("画像が読み込めませんでした\n");

		return;
	}

	if (source.channels() == 4)
		cvtColor(source, source, COLOR_BGRA2BGR);

	//! [Prepare blob]
	Mat inputBlob = blobFromImage(source, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
	//! [Prepare blob]

	//! [Set input blob]
	net.setInput(inputBlob, "data");                   //set the network input
	//! [Set input blob]

	//! [Make forward pass]
	Mat detectionMat = net.forward("detection_out");   //compute output
	//! [Make forward pass]

	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
	cv::putText(source, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

	float confidenceThreshold = 0.3;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);

		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

		if (confidence > confidenceThreshold)
		{
			float x = detectionMat.at<float>(i, 0);
			float y = detectionMat.at<float>(i, 1);
			float width = detectionMat.at<float>(i, 2);
			float height = detectionMat.at<float>(i, 3);
			int xLeftBottom = static_cast<int>((x - width / 2) * source.cols);
			int yLeftBottom = static_cast<int>((y - height / 2) * source.rows);
			int xRightTop = static_cast<int>((x + width / 2) * source.cols);
			int yRightTop = static_cast<int>((y + height / 2) * source.rows);

			Rect object(xLeftBottom, yLeftBottom,
				xRightTop - xLeftBottom,
				yRightTop - yLeftBottom);

			cv::rectangle(source, object, Scalar(0, 255, 0));

			if (objectClass < classNamesVec.size())
			{
				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = String(classNamesVec[objectClass]) + ": " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				cv::rectangle(source, Rect(Point(xLeftBottom, yLeftBottom),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				cv::putText(source, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
			else
			{
				cout << "Class: " << objectClass << endl;
				cout << "Confidence: " << confidence << endl;
				cout << " " << xLeftBottom
					<< " " << yLeftBottom
					<< " " << xRightTop
					<< " " << yRightTop << endl;
			}

			if (objectClass == 0) {
				//hantei1= String(classNamesVec[objectClass]);
				normal = 1;
				Pnormal = confidence;
			}
			if (objectClass == 1) {
				//hantei2 = String(classNamesVec[objectClass]);
				abnormal = 1;
				Pabnormal = confidence;
			}
		}
	}


	sprintf_s(outputCsvName, 256, "G:\\マイドライブ\\result_mae/result-%s.csv", sourceFileName);	//出力CSVファイル名

	if ((csv_error = fopen_s(&fp, outputCsvName, "w")) != 0) { // GPS_sakurai (3) : GPSを使った判定用CSVファイル名
		printf("hantei file open error!!\n");
		exit(EXIT_FAILURE);	/* (3)エラーの場合は通常、異常終了する 
	}
	std::fprintf(fp, "正常,識別率,異常,識別率\n");
	std::fprintf(fp, "%d,%lf,%d,%lf", normal, Pnormal, abnormal, Pabnormal);
	std::fclose(fp);
	normal = 0;
	abnormal = 0;
	Pnormal = 0;
	Pabnormal = 0;
	
	return;
} // main
*/



int main(int argc, char** argv)
{

    std::vector<std::string> class_list = load_class_list();

	/*
    cv::Mat frame;
    cv::VideoCapture capture("sample.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }
	*/
    

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    is_cuda = true;
    std::cout << is_cuda << std::endl;
	
	Mat img = imread("test.png");
    

    cv::dnn::Net net;
    load_net(net, is_cuda);

	std::vector<Detection> output;
	detect(img, net, output, class_list);

	int detections = output.size();

	for (int i = 0; i < detections; ++i)
	{

		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;
		const auto color = colors[classId % colors.size()];

		auto confidence = detection.confidence;

		cv::rectangle(img, box, color, 3);

		cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
		cv::putText(img, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		//std::cout << classId << std::endl;
		std::cout << confidence << std::endl;
	}

	cv::imshow("output", img);
	cv::waitKey(0);
	
	

	/*
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;
	
	
    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];

            auto confidence = detection.confidence;

            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            //std::cout << classId << std::endl;
            std::cout << confidence << std::endl;
        }

        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    std::cout << "Total frames: " << total_frames << "\n";
	*/
    return 0;
	
}
