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



void Evaluation_image(std::vector<WCHAR> fileName)
{
	FILE* fp;
	errno_t csv_error;
	string sourceFileName;
	char FileName[256];
	char outputCsvName[256];

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
	sourceFileName = ConvertFromUnicode(&fileName[0]);
	sprintf_s(FileName, 256, "G:\\マイドライブ\\image_mae/%s", sourceFileName);
	Mat img = imread(FileName);
	if (img.empty()) {
		printf("画像が読み込めませんでした\n");

		return;
	}
    

    //bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    bool is_cuda = true;
    //std::cout << is_cuda << std::endl;
	
	//Mat img = imread("test.png");
    

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

	return;
	
	

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
    
	
}


// エラーの表示
static void ShowError(LPCTSTR msg)
{
	DWORD errcode = GetLastError();
	std::wprintf(_T("%s errorcode: %lx\r\n"), msg, errcode);
}

// キー入力のチェック
static inline bool CheckQuitKey()
{
	return _kbhit() && (_getch() == 'q');
}

// メインエントリ
int main(int argc, char** argv)
{
	// コンソール出力を日本語可能に
	setlocale(LC_ALL, "");

	// オプション引数の値を保持する
	LPCTSTR pDir = _T("C:\\Dropbox");
	size_t bufsiz = 0;
	int waittime = 0;
	bool hasError = false;


	//if (argc == 2) {
	//	sprintf_s(modelBinary, 256, "yolov2-tiny_yamamoto_%s.weights", argv[1]);
	//}
	// 引数の解析
	if (argc > 2) {
		_TCHAR** pArg = (_TCHAR**)argv[2];
		while (*pArg) {
			if (_tcsicmp(_T("/b"), *pArg) == 0) {
				// バッファサイズ
				pArg++;
				if (*pArg) {
					bufsiz = _ttol(*pArg);
				}

			}
			else if (_tcsicmp(_T("/w"), *pArg) == 0) {
				// ウェイト時間
				pArg++;
				if (*pArg) {
					waittime = _ttoi(*pArg);
				}

			}
			else if (**pArg != '/') {
				// 監視先ディレクトリ
				pDir = *pArg;
				break;

			}
			else {
				_ftprintf(stderr, _T("不明な引数: %s\r\n"), *pArg);
				hasError = true;
			}

			pArg++;
		}

	}

	if (hasError) {
		return 2;
	}

	if (bufsiz <= 0) {
		bufsiz = 1024 * 8;
	}
	if (waittime <= 0) {
		waittime = 0;
	}

	std::wprintf(_T("監視先: %s\r\nバッファサイズ: %ld\r\nループ毎の待ち時間: %ld\r\n"),
		pDir, bufsiz, waittime);


	// 対象のディレクトリを監視用にオープンする.
	// 共有ディレクトリ使用可、対象フォルダを削除可
	// 非同期I/O使用
	HANDLE hDir = CreateFile(
		pDir, // 監視先
		FILE_LIST_DIRECTORY,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, // ReadDirectoryChangesW用
		NULL
	);
	if (hDir == INVALID_HANDLE_VALUE) {
		ShowError(_T("CreateFileでの失敗"));
		return 1;
	}

	// 監視条件 (FindFirstChangeNotificationと同じ)
	DWORD filter =
		FILE_NOTIFY_CHANGE_FILE_NAME |  // ファイル名の変更
		FILE_NOTIFY_CHANGE_DIR_NAME |  // ディレクトリ名の変更
		FILE_NOTIFY_CHANGE_ATTRIBUTES |  // 属性の変更
		FILE_NOTIFY_CHANGE_SIZE |  // サイズの変更
		FILE_NOTIFY_CHANGE_LAST_WRITE;    // 最終書き込み日時の変更

	// 変更されたファイルのリストを記録するためのバッファ.
	// 最初のReadDirectoryChangesWの通知から次のReadDirectoryChangesWまでの
	// 間に変更されたファイルの情報を格納できるだけのサイズが必要.
	// バッファオーバーとしてもファイルに変更が発生したことは感知できるが、
	// なにが変更されたかは通知できない。
	std::vector<unsigned char> buf(bufsiz);
	void* pBuf = &buf[0];

	// 非同期I/Oの完了待機用, 手動リセットモード。
	// 変更通知のイベント発報とキャンセル完了のイベント発報の
	// 2つのイベントソースがあるためイベントの流れが予想できず
	// 自動リセットイベントにするのは危険。
	HANDLE hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// 変更通知を受け取りつづけるループ
	for (;;) {
		// Sleepを使い、イベントハンドリングが遅いケースをエミュレートする.
		// ループ2回目以降ならばSleep表示中にファイルを変更しても、
		// 変更が追跡されていることを確認できる.
		// またバッファサイズを超えるほどにたくさんのファイルを変更すると
		// バッファオーバーを確認できる。
		const int mx = waittime * 10;
		for (int idx = 0; idx < mx; idx++) {
			std::wprintf(_T("sleep... %d/%d \r"), idx + 1, mx);
			Sleep(100);
		}
		std::wprintf(_T("\r\nstart.\r\n"));

		// イベントの手動リセット
		ResetEvent(hEvent);

		// 非同期I/O
		// ループ中でのみ使用・待機するので、ここ(スタック)に置いても安全.
		OVERLAPPED olp = { 0 };
		olp.hEvent = hEvent;

		// 変更を監視する.
		// 初回呼び出し時にシステムが指定サイズでバッファを確保し、そこに変更を記録する.
		// 完了通知後もシステムは変更を追跡しており、後続のReadDirectoryChangeWの
		// 呼び出しで、前回通知後からの変更をまとめて受け取ることができる.
		// バッファがあふれた場合はサイズ0で応答が返される.
		if (!ReadDirectoryChangesW(
			hDir,   // 対象ディレクトリ
			pBuf,   // 通知を格納するバッファ
			bufsiz, // バッファサイズ
			TRUE,   // サブディレクトリを対象にするか?
			filter, // 変更通知を受け取るフィルタ
			NULL,   // (結果サイズ, 非同期なので未使用)
			&olp,   // 非同期I/Oバッファ
			NULL    // (完了ルーチン, 未使用)
		)) {
			// 開始できなかった場合のエラー
			ShowError(_T("ReadDirectoryChangesWでの失敗"));
			break;
		}

		// 完了待機ループ (qキーで途中終了)
		bool quit;
		while (!(quit = CheckQuitKey())) {
			// 変更通知まち
			DWORD waitResult = WaitForSingleObject(hEvent, 500); // 0.5秒待ち
			if (waitResult != WAIT_TIMEOUT) {
				// 変更通知があった場合 (イベントがシグナル状態になった場合)
				break;
			}
			// 待ち受け表示
			std::wprintf(_T("."));
		}
		std::wprintf(_T("\r\n"));

		if (quit) {
			// 途中終了するなら非同期I/Oも中止し、
			// Overlapped構造体をシステムが使わなくなるまで待機する必要がある.
			CancelIo(hDir);
			WaitForSingleObject(hEvent, INFINITE);
			break;
		}

		// 非同期I/Oの結果を取得する.
		DWORD retsize = 0;
		if (!GetOverlappedResult(hDir, &olp, &retsize, FALSE)) {
			// 結果取得に失敗した場合
			ShowError(_T("GetOverlappedResultでの失敗"));
			break;
		}

		// 変更通知をコンソールにダンプする.
		std::wprintf(_T("returned size=%ld\r\n"), retsize);

		if (retsize == 0) {
			// 返却サイズ、0ならばバッファオーバーを示す
			std::wprintf(_T("buffer overflow!!\r\n"));

		}
		else {
			// 最初のエントリに位置付ける
			FILE_NOTIFY_INFORMATION* pData =
				reinterpret_cast<FILE_NOTIFY_INFORMATION*>(pBuf);

			// エントリの末尾まで繰り返す
			for (;;) {
				// アクションタイプを可読文字に変換
				TCHAR pActionMsg[] = _T("Others");
				//TCHAR* pActionMsg;
				//wprintf_s(_T("UNKNOWN"), pActionMsg);

				switch (pData->Action) {
				case FILE_ACTION_ADDED:
					::_tcscpy_s(pActionMsg, 256 * sizeof(TCHAR), _T("Added"));
					//pActionMsg[] =  _T("Added") ;
					//wprintf_s(_T("Added"), pActionMsg);
					break;
				default:
					break;
					//case FILE_ACTION_REMOVED:
					//	//pActionMsg = _T("Removed");
					//	wprintf_s(_T("Removed"), pActionMsg);
					//	break;
					//case FILE_ACTION_MODIFIED:
					//	//pActionMsg = _T("Modified");
					//	wprintf_s(_T("Modified"), pActionMsg);
					//	break;
					//case FILE_ACTION_RENAMED_OLD_NAME:
					//	//pActionMsg = _T("Rename Old");
					//	wprintf_s(_T("Rename Old"), pActionMsg);
					//	break;
					//case FILE_ACTION_RENAMED_NEW_NAME:
					//	//pActionMsg = _T("Rename New");
					//	wprintf_s(_T("Rename New"), pActionMsg);
					//	break;
				}

				// ファイル名はヌル終端されていないので
				// 長さから終端をつけておく.
				DWORD lenBytes = pData->FileNameLength; // 文字数ではなく, バイト数
				std::vector<WCHAR> fileName(lenBytes / sizeof(WCHAR) + 1); // ヌル終端用に+1
				std::memcpy(&fileName[0], pData->FileName, lenBytes);

				// アクションと対象ファイルを表示.
				// (ファイル名は指定ディレクトリからの相対パスで通知される.)
				std::wprintf(_T("[%s]<%s>\r\n"), pActionMsg, &fileName[0]);

				//if (pData->Action == FILE_ACTION_ADDED)	Evaluation_image(fileName);	//Evaluation_video(fileName)

				if (pData->NextEntryOffset == 0) {
					// 次のエントリは無し
					break;
				}
				// 次のエントリの位置まで移動する. (現在アドレスからの相対バイト数)
				pData = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(
					reinterpret_cast<unsigned char*>(pData) + pData->NextEntryOffset);
			}
		}
	}

	// ハンドルの解放
	CloseHandle(hEvent);
	CloseHandle(hDir);

	return 0;
}
